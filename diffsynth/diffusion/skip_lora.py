"""Inference support for Skip-LoRA adapters used by Wan video DiTs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SkipLoRAConfig:
    rank: int = 32
    lora_alpha: float = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q", "k", "v", "o", "ffn.0", "ffn.2"]
    )
    combine_mode: str = "linear"
    carry_mode: str = "accumulate"
    normalize_skip: bool = False
    detach_across_blocks: bool = True
    detach_within_block: bool = True

    def __post_init__(self) -> None:
        if self.combine_mode not in {"add", "gated_add", "linear"}:
            raise ValueError(f"Unsupported Skip-LoRA combine mode: {self.combine_mode}")
        if self.carry_mode not in {"accumulate", "passthrough"}:
            raise ValueError(f"Unsupported Skip-LoRA carry mode: {self.carry_mode}")


def _rms_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = torch.rsqrt(value.float().square().mean(dim=-1, keepdim=True) + eps)
    return value * scale.to(value.dtype)


class SkipChain(nn.Module):
    """Forward-only rank-space residual stream shared by all adapter layers."""

    def __init__(self, config: SkipLoRAConfig):
        super().__init__()
        self.config = config
        self.num_registered_layers = 0
        self._snapshots: list[Optional[torch.Tensor]] = []

    def register_layer(self) -> int:
        index = self.num_registered_layers
        self.num_registered_layers += 1
        return index

    def reset(self) -> None:
        self._snapshots = [None] * (self.num_registered_layers + 1)

    def read(self, index: int, reference: torch.Tensor) -> torch.Tensor:
        if len(self._snapshots) != self.num_registered_layers + 1:
            self.reset()
        value = self._snapshots[index]
        if value is None:
            value = next(
                (item for item in reversed(self._snapshots[: index + 1]) if item is not None),
                None,
            )
        return torch.zeros_like(reference) if value is None else value

    def write(self, index: int, value: torch.Tensor) -> None:
        self._snapshots[index + 1] = value.detach() if self.config.detach_within_block else value

    def detach(self, index: int) -> None:
        if 0 <= index < len(self._snapshots) and self._snapshots[index] is not None:
            self._snapshots[index] = self._snapshots[index].detach()


class SkipLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: SkipLoRAConfig, chain: SkipChain):
        super().__init__()
        self.base = base
        self.config = config
        self.chain = chain
        self.call_idx = chain.register_layer()
        self.lora_A = nn.Linear(base.in_features, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scaling = config.lora_alpha / config.rank
        if config.combine_mode == "gated_add":
            self.skip_gate = nn.Parameter(torch.ones((), dtype=base.weight.dtype))
        elif config.combine_mode == "linear":
            self.skip_proj = nn.Linear(config.rank, config.rank, bias=False)
            nn.init.zeros_(self.skip_proj.weight)

    def _combine(self, local: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_skip:
            incoming = _rms_normalize(incoming)
        if self.config.combine_mode == "add":
            return local + incoming
        if self.config.combine_mode == "gated_add":
            return local + self.skip_gate.to(local.dtype) * incoming
        return local + self.skip_proj(incoming)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_output = self.base(hidden_states)
        local = self.lora_A(hidden_states)
        incoming = self.chain.read(self.call_idx, local)
        # Text and image token streams have different lengths. Context K/V
        # therefore behaves as plain LoRA and does not alter the image stream.
        if incoming.shape[1] != local.shape[1]:
            return base_output + self.lora_B(local) * self.scaling
        combined = self._combine(local, incoming)
        carried = combined if self.config.carry_mode == "accumulate" else local
        self.chain.write(self.call_idx, carried)
        return base_output + self.lora_B(combined) * self.scaling


def attach_skip_lora(
    dit: nn.Module,
    config: SkipLoRAConfig,
    target_modules: Optional[list[str]] = None,
) -> SkipChain:
    """Wrap all matching Linear layers and install stream lifecycle hooks."""
    targets = target_modules or config.target_modules
    chain = SkipChain(config)
    matches = []
    for name, module in dit.named_modules():
        if isinstance(module, nn.Linear) and any(
            name == target or name.endswith("." + target) for target in targets
        ):
            parent_name, _, attribute = name.rpartition(".")
            parent = dit.get_submodule(parent_name) if parent_name else dit
            matches.append((parent, attribute, module))
    if not matches:
        raise ValueError(f"No Linear layers matched Skip-LoRA targets: {targets}")
    for parent, attribute, base in matches:
        wrapper = SkipLoRALinear(base, config, chain).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        setattr(parent, attribute, wrapper)

    block_first_call: dict[int, int] = {}
    if hasattr(dit, "blocks"):
        for block_index, block in enumerate(dit.blocks):
            indices = [m.call_idx for m in block.modules() if isinstance(m, SkipLoRALinear)]
            if indices:
                block_first_call[block_index] = min(indices)
        if len(dit.blocks):
            dit.blocks[0].register_forward_pre_hook(
                lambda module, args, kwargs: chain.reset(), with_kwargs=True
            )
        if config.detach_across_blocks:
            for block_index, first_call in block_first_call.items():
                if block_index:
                    dit.blocks[block_index].register_forward_pre_hook(
                        lambda module, args, kwargs, index=first_call: chain.detach(index),
                        with_kwargs=True,
                    )
    dit.skip_chain = chain
    return chain
