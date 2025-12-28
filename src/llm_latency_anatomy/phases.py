# llm_latency_anatomy/phases.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from . import nvtx


class CausalLM(Protocol):
    """
    Minimal protocol for the Hugging Face causal LM methods we use.
    """

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        use_cache: bool,
        past_key_values: Any | None = None,
    ) -> Any:
        ...


@dataclass(frozen=True)
class PrefillOutput:
    """
    Output of the prefill phase.
    """

    past_key_values: Any
    next_token: torch.Tensor  # [B, 1]


@torch.inference_mode()
def prefill(model: CausalLM, input_ids: torch.Tensor) -> PrefillOutput:
    """
    Run a full forward pass over the prompt to build the KV cache.
    """
    with nvtx.range('prefill'):
        out = model(input_ids=input_ids, use_cache=True)

    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return PrefillOutput(past_key_values=past, next_token=next_token)
