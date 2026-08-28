"""Loader and PyTorch adapter for the first sequence-resident CUDA prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

from fused_megakernel import fused_megakernel_forward


SEQUENCE_LENGTH = 128
MODEL_DIMENSION = 128
NUMBER_OF_HEADS = 4
NUMBER_OF_LAYERS = 4

_ELEMENTS = SEQUENCE_LENGTH * MODEL_DIMENSION
_SCORE_ELEMENTS = NUMBER_OF_HEADS * SEQUENCE_LENGTH * SEQUENCE_LENGTH

_DEBUG_LAYOUT = (
    ("layernorm", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    ("qkv.q", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    ("qkv.k", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    ("qkv.v", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    (
        "scores",
        _SCORE_ELEMENTS,
        (NUMBER_OF_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH),
    ),
    (
        "softmax",
        _SCORE_ELEMENTS,
        (NUMBER_OF_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH),
    ),
    ("pv", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    ("output_projection", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
    ("ffn", _ELEMENTS, (SEQUENCE_LENGTH, MODEL_DIMENSION)),
)

_extension = None


def load_extension(verbose: bool = False):
    global _extension
    if _extension is None:
        root = Path(__file__).resolve().parent
        _extension = load(
            name="tttj_sequence_resident_v1",
            sources=[
                str(root / "csrc" / "sequence_resident.cpp"),
                str(root / "csrc" / "sequence_resident.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--threads=4", "-lineinfo"],
            verbose=verbose,
        )
    return _extension


def _append_linear(tensors: list[torch.Tensor], linear: nn.Linear) -> None:
    if linear.bias is None:
        raise ValueError("the CUDA prototype requires linear biases")
    tensors.extend((linear.weight.detach().reshape(-1), linear.bias.detach()))


def pack_model_weights(model: nn.Module) -> torch.Tensor:
    """Pack the fixed case-5/6 parameter layout into one contiguous tensor."""
    config = model.config
    expected = (128, 4, 128, 4, True)
    actual = (
        config.d_model,
        config.num_heads,
        config.ffn_dim,
        config.num_layers,
        config.causal,
    )
    if actual != expected:
        raise ValueError(
            "sequence-resident v1 only supports D=128, H=4, F=128, "
            f"L=4, causal=True; got {actual}"
        )

    tensors: list[torch.Tensor] = []
    for layer in model.layers:
        tensors.extend((layer.norm1.weight.detach(), layer.norm1.bias.detach()))
        _append_linear(tensors, layer.attention.q_proj)
        _append_linear(tensors, layer.attention.k_proj)
        _append_linear(tensors, layer.attention.v_proj)
        _append_linear(tensors, layer.attention.out_proj)
        tensors.extend((layer.norm2.weight.detach(), layer.norm2.bias.detach()))
        _append_linear(tensors, layer.ffn_in)
        _append_linear(tensors, layer.ffn_out)
    tensors.extend((model.final_norm.weight.detach(), model.final_norm.bias.detach()))
    return torch.cat(tensors).contiguous()


def unpack_debug(debug: torch.Tensor) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, count, shape in _DEBUG_LAYOUT:
        result[name] = debug[offset : offset + count].view(shape)
        offset += count
    if offset != debug.numel():
        raise ValueError(
            f"debug tensor has {debug.numel()} values, expected {offset}"
        )
    return result


class SequenceResidentTransformer(nn.Module):
    """Inference-only adapter around a compatible baseline parameter module."""

    def __init__(
        self,
        parameter_model: nn.Module,
        verbose_build: bool = False,
    ) -> None:
        super().__init__()
        self.parameter_model = parameter_model
        self.verbose_build = verbose_build
        self.register_buffer("packed_weights", None, persistent=False)

    def prepare(self) -> None:
        packed = pack_model_weights(self.parameter_model)
        if packed.device.type != "cuda" or packed.dtype != torch.float16:
            raise ValueError("parameters must be CUDA float16 before prepare()")
        self.packed_weights = packed

    def _ensure_prepared(self, x: torch.Tensor) -> None:
        if self.packed_weights is None:
            self.prepare()
        if self.packed_weights.device != x.device:
            raise ValueError("packed weights and input must be on the same CUDA device")

    def run(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        capture_debug: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if valid_token_mask is None:
            valid_token_mask = torch.ones(
                x.shape[:2], device=x.device, dtype=torch.bool
            )
        self._ensure_prepared(x)
        if not capture_debug:
            return (
                fused_megakernel_forward(
                    x, valid_token_mask, self.packed_weights
                ),
                None,
            )
        output, debug = load_extension().forward(
            x.contiguous(),
            valid_token_mask.contiguous(),
            self.packed_weights,
            capture_debug,
        )
        return output, unpack_debug(debug) if capture_debug else None

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output, _ = self.run(x, valid_token_mask, capture_debug=False)
        return output
