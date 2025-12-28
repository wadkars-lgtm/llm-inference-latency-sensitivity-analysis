# llm_latency_anatomy/collapse.py
from __future__ import annotations

from dataclasses import dataclass

from .metrics import AggregatedMetrics


@dataclass(frozen=True)
class CollapseResult:
    """
    Collapse detection output between consecutive sweep points.
    """

    flag: str
    prefill_ratio: float | None
    decode_ratio: float | None


def detect(
    prev: AggregatedMetrics | None, cur: AggregatedMetrics, *, collapse_factor: float
) -> CollapseResult:
    """
    Detect collapse between previous and current sweep points.

    Policy (batch-safe):
      - PREFILL collapse if cur.prefill_ms_per_token_p50 / prev.prefill_ms_per_token_p50 >= collapse_factor
      - DECODE collapse if cur.decode_ms_per_tok_p50 / prev.decode_ms_per_tok_p50 >= collapse_factor

    Why:
      - Using total prefill_ms is not batch-safe (larger batches naturally take longer).
      - Using per-token prefill time is a better indicator of a regime change (OOM, fast-path loss, etc.).
    """
    if prev is None:
        return CollapseResult(flag='', prefill_ratio=None, decode_ratio=None)

    flags: list[str] = []
    pre_ratio: float | None = None
    dec_ratio: float | None = None

    if prev.prefill_ms_per_token_p50 > 0:
        pre_ratio = cur.prefill_ms_per_token_p50 / prev.prefill_ms_per_token_p50
        if pre_ratio >= collapse_factor:
            flags.append(f'COLLAPSE_PREFILL(x{pre_ratio:.1f})')

    if prev.decode_ms_per_tok_p50 > 0:
        dec_ratio = cur.decode_ms_per_tok_p50 / prev.decode_ms_per_tok_p50
        if dec_ratio >= collapse_factor:
            flags.append(f'COLLAPSE_DECODE(x{dec_ratio:.1f})')

    return CollapseResult(flag=' '.join(flags), prefill_ratio=pre_ratio, decode_ratio=dec_ratio)
