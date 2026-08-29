#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> sequence_resident_cuda(
    torch::Tensor input,
    torch::Tensor valid_mask,
    torch::Tensor packed_weights,
    bool capture_debug);

std::vector<torch::Tensor> sequence_resident(
    torch::Tensor input,
    torch::Tensor valid_mask,
    torch::Tensor packed_weights,
    bool capture_debug) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(valid_mask.is_cuda(), "valid_mask must be a CUDA tensor");
  TORCH_CHECK(packed_weights.is_cuda(), "packed_weights must be a CUDA tensor");
  return sequence_resident_cuda(
      input, valid_mask, packed_weights, capture_debug);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward",
      &sequence_resident,
      "One-CTA sequence-resident Transformer forward (CUDA)",
      py::arg("input"),
      py::arg("valid_mask"),
      py::arg("packed_weights"),
      py::arg("capture_debug") = false);
}
