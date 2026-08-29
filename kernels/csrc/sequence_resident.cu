#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

constexpr int kSequence = 128;
constexpr int kModel = 128;
constexpr int kHeads = 4;
constexpr int kHeadDim = 32;
constexpr int kLayers = 4;
constexpr int kElements = kSequence * kModel;
constexpr int kScoreElements = kHeads * kSequence * kSequence;
constexpr int kThreads = 256;
constexpr int kSharedHalfs = 5 * kElements;
constexpr int kSharedBytes = kSharedHalfs * sizeof(__half);

// 2 LayerNorm vectors, Q/K/V/O and two FFN matrices plus their biases.
constexpr int kLayerStride =
    4 * kModel + 6 * (kModel * kModel + kModel);
constexpr int kPackedElements = kLayers * kLayerStride + 2 * kModel;

constexpr int kDebugNorm = 0;
constexpr int kDebugQ = kDebugNorm + kElements;
constexpr int kDebugK = kDebugQ + kElements;
constexpr int kDebugV = kDebugK + kElements;
constexpr int kDebugScores = kDebugV + kElements;
constexpr int kDebugProbs = kDebugScores + kScoreElements;
constexpr int kDebugPv = kDebugProbs + kScoreElements;
constexpr int kDebugOut = kDebugPv + kElements;
constexpr int kDebugFfn = kDebugOut + kElements;
constexpr int kDebugElements = kDebugFfn + kElements;

struct LayerWeights {
  const __half* norm1_weight;
  const __half* norm1_bias;
  const __half* q_weight;
  const __half* q_bias;
  const __half* k_weight;
  const __half* k_bias;
  const __half* v_weight;
  const __half* v_bias;
  const __half* out_weight;
  const __half* out_bias;
  const __half* norm2_weight;
  const __half* norm2_bias;
  const __half* ffn_in_weight;
  const __half* ffn_in_bias;
  const __half* ffn_out_weight;
  const __half* ffn_out_bias;
};

__device__ __forceinline__ const __half* take(
    const __half*& cursor, int count) {
  const __half* result = cursor;
  cursor += count;
  return result;
}

__device__ __forceinline__ LayerWeights layer_weights(
    const __half* packed, int layer) {
  const __half* cursor = packed + layer * kLayerStride;
  LayerWeights weights;
  weights.norm1_weight = take(cursor, kModel);
  weights.norm1_bias = take(cursor, kModel);
  weights.q_weight = take(cursor, kElements);
  weights.q_bias = take(cursor, kModel);
  weights.k_weight = take(cursor, kElements);
  weights.k_bias = take(cursor, kModel);
  weights.v_weight = take(cursor, kElements);
  weights.v_bias = take(cursor, kModel);
  weights.out_weight = take(cursor, kElements);
  weights.out_bias = take(cursor, kModel);
  weights.norm2_weight = take(cursor, kModel);
  weights.norm2_bias = take(cursor, kModel);
  weights.ffn_in_weight = take(cursor, kElements);
  weights.ffn_in_bias = take(cursor, kModel);
  weights.ffn_out_weight = take(cursor, kElements);
  weights.ffn_out_bias = take(cursor, kModel);
  return weights;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ void layer_norm(
    const __half* input,
    __half* output,
    const __half* weight,
    const __half* bias) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int row = warp; row < kSequence; row += kThreads / 32) {
    float local_sum = 0.0f;
    float local_square_sum = 0.0f;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane + item * 32;
      const float value = __half2float(input[row * kModel + column]);
      local_sum += value;
      local_square_sum = fmaf(value, value, local_square_sum);
    }
    const float mean = warp_sum(local_sum) / static_cast<float>(kModel);
    const float mean_square =
        warp_sum(local_square_sum) / static_cast<float>(kModel);
    const float inverse_std = rsqrtf(fmaxf(mean_square - mean * mean, 0.0f) + 1.0e-5f);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int column = lane + item * 32;
      const int index = row * kModel + column;
      const float normalized =
          (__half2float(input[index]) - mean) * inverse_std;
      const float affine = fmaf(
          normalized, __half2float(weight[column]), __half2float(bias[column]));
      output[index] = __float2half_rn(affine);
    }
  }
}

__device__ void linear(
    const __half* input,
    __half* output,
    const __half* weight,
    const __half* bias) {
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int row = index / kModel;
    const int column = index - row * kModel;
    float accumulator = __half2float(bias[column]);
#pragma unroll 4
    for (int reduction = 0; reduction < kModel; ++reduction) {
      accumulator = fmaf(
          __half2float(input[row * kModel + reduction]),
          __half2float(weight[column * kModel + reduction]),
          accumulator);
    }
    output[index] = __float2half_rn(accumulator);
  }
}

__device__ void attention_scores(
    const __half* q,
    const __half* k,
    __half* scores,
    const bool* valid,
    int head) {
  constexpr float scale = 0.1767766952966369f;
  for (int index = threadIdx.x;
       index < kSequence * kSequence;
       index += blockDim.x) {
    const int query = index / kSequence;
    const int key = index - query * kSequence;
    if (key > query || !valid[key]) {
      scores[index] = __float2half_rn(-INFINITY);
      continue;
    }
    float accumulator = 0.0f;
    const int q_base = query * kModel + head * kHeadDim;
    const int k_base = key * kModel + head * kHeadDim;
#pragma unroll
    for (int reduction = 0; reduction < kHeadDim; ++reduction) {
      accumulator = fmaf(
          __half2float(q[q_base + reduction]),
          __half2float(k[k_base + reduction]),
          accumulator);
    }
    scores[index] = __float2half_rn(accumulator * scale);
  }
}

__device__ void attention_softmax(__half* scores) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int row = warp; row < kSequence; row += kThreads / 32) {
    float local_max = -INFINITY;
    for (int column = lane; column < kSequence; column += 32) {
      local_max = fmaxf(
          local_max, __half2float(scores[row * kSequence + column]));
    }
    const float maximum = warp_max(local_max);
    float local_sum = 0.0f;
    for (int column = lane; column < kSequence; column += 32) {
      const float value = __half2float(scores[row * kSequence + column]);
      local_sum += expf(value - maximum);
    }
    const float denominator = warp_sum(local_sum);
    for (int column = lane; column < kSequence; column += 32) {
      const float value = __half2float(scores[row * kSequence + column]);
      scores[row * kSequence + column] =
          __float2half_rn(expf(value - maximum) / denominator);
    }
  }
}

__device__ void attention_pv(
    const __half* probs,
    const __half* v,
    __half* context,
    int head) {
  constexpr int head_elements = kSequence * kHeadDim;
  for (int index = threadIdx.x; index < head_elements; index += blockDim.x) {
    const int query = index / kHeadDim;
    const int column = index - query * kHeadDim;
    float accumulator = 0.0f;
    for (int key = 0; key < kSequence; ++key) {
      accumulator = fmaf(
          __half2float(probs[query * kSequence + key]),
          __half2float(v[key * kModel + head * kHeadDim + column]),
          accumulator);
    }
    context[query * kModel + head * kHeadDim + column] =
        __float2half_rn(accumulator);
  }
}

__device__ void copy_debug(
    const __half* source, __half* debug, int offset, int count) {
  if (debug == nullptr) {
    return;
  }
  for (int index = threadIdx.x; index < count; index += blockDim.x) {
    debug[offset + index] = source[index];
  }
}

__device__ void linear_residual(
    const __half* input,
    const __half* residual,
    __half* output,
    const __half* weight,
    const __half* bias,
    const bool* valid,
    __half* debug,
    int debug_offset) {
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int row = index / kModel;
    const int column = index - row * kModel;
    float branch = __half2float(bias[column]);
#pragma unroll 4
    for (int reduction = 0; reduction < kModel; ++reduction) {
      branch = fmaf(
          __half2float(input[row * kModel + reduction]),
          __half2float(weight[column * kModel + reduction]),
          branch);
    }
    const __half rounded_branch = valid[row]
        ? __float2half_rn(branch)
        : __float2half_rn(0.0f);
    if (debug != nullptr) {
      debug[debug_offset + index] = rounded_branch;
    }
    output[index] = __float2half_rn(
        __half2float(residual[index]) + __half2float(rounded_branch));
  }
}

__device__ void ffn_gelu(
    const __half* input,
    __half* output,
    const __half* weight,
    const __half* bias) {
  constexpr float inverse_sqrt_two = 0.7071067811865475f;
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int row = index / kModel;
    const int column = index - row * kModel;
    float accumulator = __half2float(bias[column]);
#pragma unroll 4
    for (int reduction = 0; reduction < kModel; ++reduction) {
      accumulator = fmaf(
          __half2float(input[row * kModel + reduction]),
          __half2float(weight[column * kModel + reduction]),
          accumulator);
    }
    const __half rounded_linear = __float2half_rn(accumulator);
    const float value = __half2float(rounded_linear);
    const float gelu = 0.5f * value * (1.0f + erff(value * inverse_sqrt_two));
    output[index] = __float2half_rn(gelu);
  }
}

__device__ void ffn_residual(
    const __half* input,
    const __half* residual,
    __half* output,
    const __half* weight,
    const __half* bias,
    const bool* valid,
    __half* debug) {
  for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
    const int row = index / kModel;
    const int column = index - row * kModel;
    float branch = __half2float(bias[column]);
#pragma unroll 4
    for (int reduction = 0; reduction < kModel; ++reduction) {
      branch = fmaf(
          __half2float(input[row * kModel + reduction]),
          __half2float(weight[column * kModel + reduction]),
          branch);
    }
    const __half rounded_branch = __float2half_rn(branch);
    if (debug != nullptr) {
      debug[kDebugFfn + index] = rounded_branch;
    }
    output[index] = valid[row]
        ? __float2half_rn(
              __half2float(residual[index]) + __half2float(rounded_branch))
        : __float2half_rn(0.0f);
  }
}

__global__ void sequence_resident_kernel(
    const __half* input,
    const bool* valid_mask,
    const __half* packed,
    __half* output,
    int batch_size,
    int* sequence_counter,
    __half* debug) {
  extern __shared__ __half shared[];
  __half* x = shared;
  __half* norm = x + kElements;
  __half* q = norm + kElements;
  __half* k = q + kElements;
  __half* v = k + kElements;

  while (true) {
    __shared__ int shared_sequence;
    if (threadIdx.x == 0) {
      shared_sequence = atomicAdd(sequence_counter, 1);
    }
    __syncthreads();
    const int sequence = shared_sequence;
    if (sequence >= batch_size) {
      return;
    }

    const __half* sequence_input = input + sequence * kElements;
    const bool* valid = valid_mask + sequence * kSequence;
    __half* sequence_output = output + sequence * kElements;
    __half* sequence_debug = sequence == 0 ? debug : nullptr;

    for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
      x[index] = sequence_input[index];
    }
    __syncthreads();

    for (int layer = 0; layer < kLayers; ++layer) {
      const LayerWeights weights = layer_weights(packed, layer);
      layer_norm(
          x, norm, weights.norm1_weight, weights.norm1_bias);
      __syncthreads();
      if (layer == 0) {
        copy_debug(norm, sequence_debug, kDebugNorm, kElements);
      }
      __syncthreads();

      linear(norm, q, weights.q_weight, weights.q_bias);
      linear(norm, k, weights.k_weight, weights.k_bias);
      linear(norm, v, weights.v_weight, weights.v_bias);
      __syncthreads();
      if (layer == 0) {
        copy_debug(q, sequence_debug, kDebugQ, kElements);
        copy_debug(k, sequence_debug, kDebugK, kElements);
        copy_debug(v, sequence_debug, kDebugV, kElements);
      }
      __syncthreads();

      for (int head = 0; head < kHeads; ++head) {
        attention_scores(q, k, norm, valid, head);
        __syncthreads();
        if (layer == 0 && sequence_debug != nullptr) {
          copy_debug(
              norm,
              sequence_debug,
              kDebugScores + head * kSequence * kSequence,
              kSequence * kSequence);
        }
        __syncthreads();
        attention_softmax(norm);
        __syncthreads();
        if (layer == 0 && sequence_debug != nullptr) {
          copy_debug(
              norm,
              sequence_debug,
              kDebugProbs + head * kSequence * kSequence,
              kSequence * kSequence);
        }
        __syncthreads();
        attention_pv(norm, v, q, head);
        __syncthreads();
      }
      if (layer == 0) {
        copy_debug(q, sequence_debug, kDebugPv, kElements);
      }
      __syncthreads();

      linear_residual(
          q,
          x,
          norm,
          weights.out_weight,
          weights.out_bias,
          valid,
          layer == 0 ? sequence_debug : nullptr,
          kDebugOut);
      __syncthreads();
      for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
        x[index] = norm[index];
      }
      __syncthreads();

      layer_norm(
          x, norm, weights.norm2_weight, weights.norm2_bias);
      __syncthreads();
      ffn_gelu(
          norm, q, weights.ffn_in_weight, weights.ffn_in_bias);
      __syncthreads();
      ffn_residual(
          q,
          x,
          norm,
          weights.ffn_out_weight,
          weights.ffn_out_bias,
          valid,
          layer == 0 ? sequence_debug : nullptr);
      __syncthreads();
      for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
        x[index] = norm[index];
      }
      __syncthreads();
    }

    const __half* final_norm_weight = packed + kLayers * kLayerStride;
    const __half* final_norm_bias = final_norm_weight + kModel;
    layer_norm(x, norm, final_norm_weight, final_norm_bias);
    __syncthreads();
    for (int index = threadIdx.x; index < kElements; index += blockDim.x) {
      const int row = index / kModel;
      sequence_output[index] = valid[row]
          ? norm[index]
          : __float2half_rn(0.0f);
    }
    __syncthreads();
  }
}

}  // namespace

std::vector<torch::Tensor> sequence_resident_cuda(
    torch::Tensor input,
    torch::Tensor valid_mask,
    torch::Tensor packed_weights,
    bool capture_debug) {
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must be float16");
  TORCH_CHECK(valid_mask.scalar_type() == at::kBool, "valid_mask must be bool");
  TORCH_CHECK(
      packed_weights.scalar_type() == at::kHalf,
      "packed_weights must be float16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(valid_mask.is_contiguous(), "valid_mask must be contiguous");
  TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must have shape [B, 128, 128]");
  TORCH_CHECK(
      input.size(1) == kSequence && input.size(2) == kModel,
      "only sequence length 128 and model dimension 128 are supported");
  TORCH_CHECK(
      valid_mask.dim() == 2 &&
          valid_mask.size(0) == input.size(0) &&
          valid_mask.size(1) == kSequence,
      "valid_mask must have shape [B, 128]");
  TORCH_CHECK(
      packed_weights.numel() == kPackedElements,
      "packed_weights has the wrong number of elements");
  TORCH_CHECK(
      input.get_device() == valid_mask.get_device() &&
          input.get_device() == packed_weights.get_device(),
      "all tensors must be on the same CUDA device");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  auto counter = torch::empty(
      {1}, input.options().dtype(torch::kInt32));
  auto debug = capture_debug
      ? torch::empty({kDebugElements}, input.options())
      : torch::empty({0}, input.options());

  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  C10_CUDA_CHECK(cudaMemsetAsync(
      counter.data_ptr<int>(), 0, sizeof(int), stream.stream()));

  static std::once_flag attribute_once;
  std::call_once(attribute_once, []() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sequence_resident_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes));
  });

  const cudaDeviceProp* properties = at::cuda::getDeviceProperties(
      input.get_device());
  const int blocks = properties->multiProcessorCount;
  sequence_resident_kernel<<<blocks, kThreads, kSharedBytes, stream.stream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      valid_mask.data_ptr<bool>(),
      reinterpret_cast<const __half*>(packed_weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      static_cast<int>(input.size(0)),
      counter.data_ptr<int>(),
      capture_debug
          ? reinterpret_cast<__half*>(debug.data_ptr<at::Half>())
          : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, debug};
}
