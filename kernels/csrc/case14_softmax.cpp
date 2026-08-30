#include <torch/extension.h>

torch::Tensor case14_softmax_cuda(
    torch::Tensor input,
    int64_t logical_classes,
    int64_t query_start,
    int64_t query_rows,
    double input_scale,
    bool inplace,
    bool fast_exp);
torch::Tensor case14_softmax_stats_cuda(
    torch::Tensor input,
    int64_t logical_classes,
    int64_t query_start,
    int64_t query_rows,
    double input_scale);

void check_softmax_input(
    const torch::Tensor& input,
    int64_t logical_classes,
    int64_t query_start,
    int64_t query_rows,
    double input_scale) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be FP16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() >= 2, "input must have at least two dimensions");
  TORCH_CHECK(
      logical_classes >= input.size(-1),
      "logical width must cover the stored input width");
  TORCH_CHECK(
      query_start < 0 || (query_start >= 0 && query_rows > 0),
      "query_rows must be positive when causal masking is enabled");
  TORCH_CHECK(input_scale > 0.0, "input scale must be positive");
}

torch::Tensor case14_softmax(
    torch::Tensor input,
    int64_t logical_classes,
    int64_t query_start,
    int64_t query_rows,
    double input_scale,
    bool inplace,
    bool fast_exp) {
  check_softmax_input(
      input, logical_classes, query_start, query_rows, input_scale);
  return case14_softmax_cuda(
      input,
      logical_classes,
      query_start,
      query_rows,
      input_scale,
      inplace,
      fast_exp);
}

torch::Tensor case14_softmax_stats(
    torch::Tensor input,
    int64_t logical_classes,
    int64_t query_start,
    int64_t query_rows,
    double input_scale) {
  check_softmax_input(
      input, logical_classes, query_start, query_rows, input_scale);
  return case14_softmax_stats_cuda(
      input, logical_classes, query_start, query_rows, input_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &case14_softmax,
      "Case-14 exact-order softmax (CUDA)",
      py::arg("input"),
      py::arg("logical_classes"),
      py::arg("query_start") = -1,
      py::arg("query_rows") = 0,
      py::arg("input_scale") = 1.0,
      py::arg("inplace") = false,
      py::arg("fast_exp") = false);
  module.def(
      "stats",
      &case14_softmax_stats,
      "Case-14 exact-order softmax row statistics (CUDA)",
      py::arg("input"),
      py::arg("logical_classes"),
      py::arg("query_start") = -1,
      py::arg("query_rows") = 0,
      py::arg("input_scale") = 1.0);
}
