# llm_latency_anatomy/nvtx.py
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def _enabled() -> bool:
    """
    Whether NVTX ranges should be emitted.
    """
    return torch.cuda.is_available()


@contextmanager
def range(name: str) -> Iterator[None]:
    """
    Create an NVTX range for Nsight Systems.
    """
    if _enabled():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if _enabled():
            torch.cuda.nvtx.range_pop()


def mark(name: str) -> None:
    """
    Emit an instantaneous NVTX marker.
    """
    if _enabled():
        torch.cuda.nvtx.mark(name)
