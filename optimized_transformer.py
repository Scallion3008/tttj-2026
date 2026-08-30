"""Public construction API for every implemented optimized transformer case."""

from __future__ import annotations

from typing import Final

import torch.nn as nn

from kernels.case13_hybrid import (
    Case13OptimizedTransformer,
    GraphedCase13Hybrid,
)
from kernels.layerwise_hybrid import Case8OptimizedTransformer
from kernels.sequence_resident import SequenceResidentTransformer


# Values are (batch, sequence, model dimension, heads). All implemented cases
# use F=D, four layers, and causal attention.
IMPLEMENTED_CASES: Final[dict[int, tuple[int, int, int, int]]] = {
    1: (64, 128, 128, 4),
    2: (1, 128, 128, 4),
    3: (4, 128, 128, 4),
    4: (16, 128, 128, 4),
    5: (128, 128, 128, 4),
    6: (10000, 128, 128, 4),
    7: (64, 128, 32, 4),
    8: (64, 128, 1024, 4),
    9: (64, 128, 128, 1),
    10: (64, 128, 128, 2),
    11: (64, 128, 128, 16),
    12: (64, 32, 128, 4),
    13: (64, 1024, 128, 4),
}


def case_number_for_model(parameter_model: nn.Module) -> int:
    """Return the implemented benchmark case matching ``parameter_model``."""
    config = parameter_model.config
    fixed = (
        config.ffn_dim == config.d_model
        and config.num_layers == 4
        and config.causal
    )
    shape = (
        config.batch_size,
        config.seq_len,
        config.d_model,
        config.num_heads,
    )
    if fixed:
        for case_number, implemented_shape in IMPLEMENTED_CASES.items():
            if shape == implemented_shape:
                return case_number
    raise ValueError(
        "no optimized transformer is implemented for "
        f"shape={shape}, ffn_dim={config.ffn_dim}, "
        f"layers={config.num_layers}, causal={config.causal}"
    )


def make_optimized_transformer(
    parameter_model: nn.Module,
    *,
    verbose_build: bool = False,
) -> nn.Module:
    """Build and prepare the fastest implementation for a supported case.

    ``parameter_model`` supplies the weights and must already be on its target
    CUDA device in FP16, matching the organizer benchmark model contract.
    """
    case_number = case_number_for_model(parameter_model)
    if case_number == 8:
        return Case8OptimizedTransformer(parameter_model)
    if case_number == 13:
        return GraphedCase13Hybrid(
            Case13OptimizedTransformer(parameter_model)
        )
    optimized = SequenceResidentTransformer(
        parameter_model,
        verbose_build=verbose_build,
    )
    optimized.prepare()
    return optimized
