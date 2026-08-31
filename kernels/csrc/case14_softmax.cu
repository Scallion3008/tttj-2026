#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 1024;
constexpr int kWarp = 32;
constexpr int kIlp = 4;

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffff, value, offset);
    value = value < other ? other : value;
  }
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_max(float value, float* shared) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
  value = warp_max(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarp
      ? shared[lane]
      : -std::numeric_limits<float>::infinity();
  if (warp == 0) {
    value = warp_max(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
  const int lane = threadIdx.x & (kWarp - 1);
  const int warp = threadIdx.x / kWarp;
  value = warp_sum(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < kWarp ? shared[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

// CUDA's correctly rounded FP32 division uses an approximate reciprocal,
// one Newton refinement of that reciprocal, then a quotient/remainder
// correction. The reciprocal depends only on the row sum, so compute that
// invariant part once per thread and retain the same correction per element.
__device__ __forceinline__ float refined_reciprocal(float divisor) {
  float reciprocal;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(divisor));
  const float error = fmaf(-divisor, reciprocal, 1.0f);
  return fmaf(reciprocal, error, reciprocal);
}

__device__ __forceinline__ float divide_with_reciprocal(
    float numerator,
    float divisor,
    float reciprocal) {
  const float quotient = numerator * reciprocal;
  const float remainder = fmaf(-divisor, quotient, numerator);
  return fmaf(reciprocal, remainder, quotient);
}

__device__ __forceinline__ float scaled_half_to_float(
    __half value,
    __half scale) {
  return __half2float(__hmul(value, scale));
}

struct __align__(8) Half4 {
  __half2 low;
  __half2 high;
};

template <bool Vectorized>
__device__ __forceinline__ void load_scaled_half4(
    const __half* input,
    int offset,
    int stored_classes,
    __half scale,
    float values[kIlp]) {
  if constexpr (Vectorized) {
    const Half4 packed = *reinterpret_cast<const Half4*>(input + offset);
    const __half2 scale2 = __halves2half2(scale, scale);
    const float2 low = __half22float2(__hmul2(packed.low, scale2));
    const float2 high = __half22float2(__hmul2(packed.high, scale2));
    values[0] = low.x;
    values[1] = low.y;
    values[2] = high.x;
    values[3] = high.y;
  } else {
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      values[item] = index < stored_classes
          ? scaled_half_to_float(input[index], scale)
          : -std::numeric_limits<float>::infinity();
    }
  }
}

template <bool Vectorized>
__device__ __forceinline__ void store_half4(
    __half* output,
    int offset,
    int stored_classes,
    const float values[kIlp]) {
  if constexpr (Vectorized) {
    Half4 packed;
    packed.low = __floats2half2_rn(values[0], values[1]);
    packed.high = __floats2half2_rn(values[2], values[3]);
    *reinterpret_cast<Half4*>(output + offset) = packed;
  } else {
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      if (offset + item < stored_classes) {
        output[offset + item] = __float2half_rn(values[item]);
      }
    }
  }
}

// This follows cunn_SoftMaxForward<4, float, float, float> from the pinned
// PyTorch build. The input conversion and output conversion are folded into
// the loads/stores; the FP32 reduction order and exp/divide epilogue are kept.
template <bool FastExp, bool Vectorized>
__global__ void exact_softmax_kernel(
    const __half* input,
    __half* output,
    int stored_classes,
    int logical_classes,
    int query_start,
    int query_rows,
    float input_scale) {
  __shared__ float reduction[32];
  const int64_t row = blockIdx.x;
  input += row * static_cast<int64_t>(stored_classes);
  output += row * static_cast<int64_t>(stored_classes);
  const int valid_classes = query_start < 0
      ? stored_classes
      : min(stored_classes, query_start + static_cast<int>(row % query_rows) + 1);
  const __half half_input_scale = __float2half_rn(input_scale);

  const int last = logical_classes % (kIlp * kThreads);
  const int vectorized_end = logical_classes - last;
  const int stored_vectorized_end = min(stored_classes, vectorized_end);
  float thread_max = -std::numeric_limits<float>::infinity();
  int vector_offset = threadIdx.x;
#pragma unroll 8
  for (; vector_offset * kIlp < stored_vectorized_end;
       vector_offset += kThreads) {
    const int offset = vector_offset * kIlp;
    float values[kIlp];
    load_scaled_half4<Vectorized>(
        input, offset, stored_classes, half_input_scale, values);
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      const float value = index < valid_classes
          ? values[item]
          : -std::numeric_limits<float>::infinity();
      thread_max = thread_max < value ? value : thread_max;
    }
  }
  for (int offset = vectorized_end + threadIdx.x;
       offset < valid_classes;
       offset += kThreads) {
    const float value = scaled_half_to_float(input[offset], half_input_scale);
    thread_max = thread_max < value ? value : thread_max;
  }
  const float max_value = block_max(thread_max, reduction);

  float thread_sum = 0.0f;
  vector_offset = threadIdx.x;
#pragma unroll 8
  for (; vector_offset * kIlp < stored_vectorized_end;
       vector_offset += kThreads) {
    const int offset = vector_offset * kIlp;
    float values[kIlp];
    load_scaled_half4<Vectorized>(
        input, offset, stored_classes, half_input_scale, values);
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      const float value = index < valid_classes
          ? values[item]
          : -std::numeric_limits<float>::infinity();
      thread_sum += FastExp ? __expf(value - max_value) : std::exp(value - max_value);
    }
  }
  for (int offset = vectorized_end + threadIdx.x;
       offset < valid_classes;
       offset += kThreads) {
    const float value = scaled_half_to_float(input[offset], half_input_scale);
    thread_sum += FastExp ? __expf(value - max_value) : std::exp(value - max_value);
  }
  const float sum = block_sum(thread_sum, reduction);
  const float reciprocal = refined_reciprocal(sum);

  vector_offset = threadIdx.x;
#pragma unroll 8
  for (; vector_offset * kIlp < stored_vectorized_end;
       vector_offset += kThreads) {
    const int offset = vector_offset * kIlp;
    float values[kIlp];
    float output_values[kIlp];
    load_scaled_half4<Vectorized>(
        input, offset, stored_classes, half_input_scale, values);
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      if (index < stored_classes) {
        const float value = index < valid_classes
            ? values[item]
            : -std::numeric_limits<float>::infinity();
        const float probability = divide_with_reciprocal(
            FastExp ? __expf(value - max_value) : std::exp(value - max_value),
            sum,
            reciprocal);
        output_values[item] = probability;
      }
    }
    store_half4<Vectorized>(output, offset, stored_classes, output_values);
  }
  for (int offset = vectorized_end + threadIdx.x;
       offset < stored_classes;
       offset += kThreads) {
    const float value = offset < valid_classes
        ? scaled_half_to_float(input[offset], half_input_scale)
        : -std::numeric_limits<float>::infinity();
    const float probability = divide_with_reciprocal(
        FastExp ? __expf(value - max_value) : std::exp(value - max_value),
        sum,
        reciprocal);
    output[offset] = __float2half_rn(probability);
  }
}

template <bool Vectorized>
__global__ void exact_softmax_stats_kernel(
    const __half* input,
    float* statistics,
    int stored_classes,
    int logical_classes,
    int query_start,
    int query_rows,
    float input_scale) {
  __shared__ float reduction[32];
  const int64_t row = blockIdx.x;
  input += row * static_cast<int64_t>(stored_classes);
  const int valid_classes = query_start < 0
      ? stored_classes
      : min(stored_classes, query_start + static_cast<int>(row % query_rows) + 1);
  const __half half_input_scale = __float2half_rn(input_scale);
  const int last = logical_classes % (kIlp * kThreads);
  const int vectorized_end = logical_classes - last;
  const int stored_vectorized_end = min(stored_classes, vectorized_end);

  float thread_max = -std::numeric_limits<float>::infinity();
  int vector_offset = threadIdx.x;
#pragma unroll 8
  for (; vector_offset * kIlp < stored_vectorized_end;
       vector_offset += kThreads) {
    const int offset = vector_offset * kIlp;
    float values[kIlp];
    load_scaled_half4<Vectorized>(
        input, offset, stored_classes, half_input_scale, values);
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      const float value = index < valid_classes
          ? values[item]
          : -std::numeric_limits<float>::infinity();
      thread_max = thread_max < value ? value : thread_max;
    }
  }
  for (int offset = vectorized_end + threadIdx.x;
       offset < valid_classes;
       offset += kThreads) {
    const float value = scaled_half_to_float(input[offset], half_input_scale);
    thread_max = thread_max < value ? value : thread_max;
  }
  const float max_value = block_max(thread_max, reduction);

  float thread_sum = 0.0f;
  vector_offset = threadIdx.x;
#pragma unroll 8
  for (; vector_offset * kIlp < stored_vectorized_end;
       vector_offset += kThreads) {
    const int offset = vector_offset * kIlp;
    float values[kIlp];
    load_scaled_half4<Vectorized>(
        input, offset, stored_classes, half_input_scale, values);
#pragma unroll
    for (int item = 0; item < kIlp; ++item) {
      const int index = offset + item;
      const float value = index < valid_classes
          ? values[item]
          : -std::numeric_limits<float>::infinity();
      thread_sum += std::exp(value - max_value);
    }
  }
  for (int offset = vectorized_end + threadIdx.x;
       offset < valid_classes;
       offset += kThreads) {
    const float value = scaled_half_to_float(input[offset], half_input_scale);
    thread_sum += std::exp(value - max_value);
  }
  const float sum = block_sum(thread_sum, reduction);
  if (threadIdx.x == 0) {
    statistics[row * 2] = max_value;
    statistics[row * 2 + 1] = sum;
  }
}

}  // namespace

torch::Tensor case14_softmax_cuda(
    torch::Tensor input,
    int64_t logical_classes64,
    int64_t query_start64,
    int64_t query_rows64,
    double input_scale64,
    bool inplace,
    bool fast_exp) {
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t stored_classes64 = input.size(-1);
  TORCH_CHECK(logical_classes64 > 1024, "case-14 softmax expects more than 1024 columns");
  TORCH_CHECK(
      logical_classes64 <= std::numeric_limits<int>::max(),
      "softmax row is too wide");
  const int64_t rows = input.numel() / stored_classes64;
  TORCH_CHECK(rows <= std::numeric_limits<unsigned int>::max(), "too many softmax rows");
  TORCH_CHECK(query_start64 <= std::numeric_limits<int>::max(), "query offset is too large");
  TORCH_CHECK(query_rows64 <= std::numeric_limits<int>::max(), "query tile is too large");
  auto output = inplace ? input : torch::empty_like(input);
  const dim3 grid(static_cast<unsigned int>(rows));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const bool vectorized = stored_classes64 % kIlp == 0;
#define LAUNCH_SOFTMAX(FAST, VECTOR)                                      \
    exact_softmax_kernel<FAST, VECTOR><<<grid, kThreads, 0, stream>>>(    \
        reinterpret_cast<const __half*>(input.const_data_ptr<at::Half>()), \
        reinterpret_cast<__half*>(output.mutable_data_ptr<at::Half>()),   \
        static_cast<int>(stored_classes64),                              \
        static_cast<int>(logical_classes64),                             \
        static_cast<int>(query_start64),                                 \
        static_cast<int>(query_rows64),                                  \
        static_cast<float>(input_scale64))
  if (fast_exp && vectorized) {
    LAUNCH_SOFTMAX(true, true);
  } else if (fast_exp) {
    LAUNCH_SOFTMAX(true, false);
  } else if (vectorized) {
    LAUNCH_SOFTMAX(false, true);
  } else {
    LAUNCH_SOFTMAX(false, false);
  }
#undef LAUNCH_SOFTMAX
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor case14_softmax_stats_cuda(
    torch::Tensor input,
    int64_t logical_classes64,
    int64_t query_start64,
    int64_t query_rows64,
    double input_scale64) {
  c10::cuda::CUDAGuard guard(input.device());
  const int64_t stored_classes64 = input.size(-1);
  TORCH_CHECK(logical_classes64 > 1024, "case-14 softmax expects more than 1024 columns");
  TORCH_CHECK(
      logical_classes64 <= std::numeric_limits<int>::max(),
      "softmax row is too wide");
  const int64_t rows = input.numel() / stored_classes64;
  TORCH_CHECK(rows <= std::numeric_limits<unsigned int>::max(), "too many softmax rows");
  auto statistics = torch::empty(
      {rows, 2}, input.options().dtype(torch::kFloat32));
  const dim3 grid(static_cast<unsigned int>(rows));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (stored_classes64 % kIlp == 0) {
    exact_softmax_stats_kernel<true><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.const_data_ptr<at::Half>()),
        statistics.mutable_data_ptr<float>(),
        static_cast<int>(stored_classes64),
        static_cast<int>(logical_classes64),
        static_cast<int>(query_start64),
        static_cast<int>(query_rows64),
        static_cast<float>(input_scale64));
  } else {
    exact_softmax_stats_kernel<false><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(input.const_data_ptr<at::Half>()),
        statistics.mutable_data_ptr<float>(),
        static_cast<int>(stored_classes64),
        static_cast<int>(logical_classes64),
        static_cast<int>(query_start64),
        static_cast<int>(query_rows64),
        static_cast<float>(input_scale64));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return statistics;
}
