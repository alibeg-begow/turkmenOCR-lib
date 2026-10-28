"""Partial CPU placement of mixture-of-experts weights for small GPUs.

Most TurkmenOCR parameters live in the routed experts of its
mixture-of-experts layers, and each token is processed by only 6 of the 64
experts in a layer. When the whole model does not fit in video memory, some
experts stay in system memory and run on the CPU. Only the small activation
tensors for the tokens routed to them cross the PCIe bus; their weights never
do, which keeps generation reasonably fast. Everything else, including the
vision encoders, attention and shared experts, stays on the GPU.

This module imports PyTorch at import time and must only be imported after
the runtime environment has been prepared.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class CpuExpert(nn.Module):
    """Runs a wrapped expert on the CPU in float32 and returns to the caller's device."""

    def __init__(self, expert: nn.Module) -> None:
        super().__init__()
        # Stored outside the module registry so model.to(device) leaves it on the CPU.
        object.__setattr__(self, "expert", expert.float().eval())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.expert(hidden_states.to("cpu", torch.float32))
        return output.to(hidden_states.device, hidden_states.dtype)


def module_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def expert_lists(model: nn.Module) -> List[nn.ModuleList]:
    return [
        module.experts
        for module in model.modules()
        if isinstance(getattr(module, "experts", None), nn.ModuleList)
    ]


def place_model(model: nn.Module, device: str, gpu_budget: int) -> Tuple[int, int]:
    """Move a CPU-loaded model to the GPU, keeping experts on the CPU as needed.

    Experts are chosen round-robin across layers, starting from the highest
    expert index, so every layer keeps most of its experts on the GPU.

    Returns:
        The number of experts kept on the CPU and the total number of experts.
    """
    excess = module_bytes(model) - gpu_budget
    lists = expert_lists(model)
    total = sum(len(experts) for experts in lists)
    width = max((len(experts) for experts in lists), default=0)
    order = [(experts, index) for index in reversed(range(width)) for experts in lists if index < len(experts)]

    offloaded = 0
    for experts, index in order:
        if excess <= 0:
            break
        excess -= module_bytes(experts[index])
        experts[index] = CpuExpert(experts[index])
        offloaded += 1
    model.to(device)
    return offloaded, total
