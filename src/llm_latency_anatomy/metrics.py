# llm_latency_anatomy/metrics.py
from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import median
from typing import Any

import torch

from . import nvtx
from .phases import CausalLM, PrefillOutput, prefill


@dataclass(frozen=True)
class RunMeasurement:
    """
    Raw measurements from a single rep.

    Fields:
      - prefill_ms: prefill wall time in ms (covers whole batch [B, L])
      - decode_step_ms: list of per-step decode times in ms (each step generates B tokens total)
      - reserved_mb_peak: peak reserved CUDA memory since last reset
      - allocated_mb_peak: peak allocated CUDA memory since last reset
    """

    prefill_ms: float
    decode_step_ms: list[float]
    reserved_mb_peak: float
    allocated_mb_peak: float


@dataclass(frozen=True)
class AggregatedMetrics:
    """
    Aggregated p50 (median) metrics across reps for one sweep point.

    Notes:
      - All "per-token" and "tok/s" metrics are for *total tokens across the batch*.
    """

    seq_len: int
    batch_size: int
    gen_tokens: int

    prefill_ms_p50: float
    prefill_ms_per_token_p50: float
    prefill_tok_per_s_p50: float

    decode_total_ms_p50: float
    decode_ms_per_tok_p50: float  # ms per decode step (per sequence token), i.e. one step producing B tokens total
    decode_tok_per_s_p50: float  # total generated tokens/s across batch

    total_tok_s: float

    vram_reserved_mb: float
    vram_allocated_mb: float


def cuda_sync() -> None:
    """
    Synchronize CUDA for accurate timing.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def mem_mb_peaks() -> tuple[float, float]:
    """
    Return (reserved_mb_peak, allocated_mb_peak) since the last reset_peak_memory_stats().
    """
    reserved = torch.cuda.max_memory_reserved() / (1024**2)
    allocated = torch.cuda.max_memory_allocated() / (1024**2)
    return reserved, allocated


@torch.inference_mode()
def measure_one(model: CausalLM, input_ids: torch.Tensor, gen_tokens: int) -> RunMeasurement:
    """
    Measure one rep: prefill + decode.

    Timing model:
      - Synchronize before/after prefill and each decode step.
      - NVTX ranges:
          - 'timed_prefill'
          - 'timed_decode_loop'
          - 'decode_step_i'
    """
    # Prefill (whole prompt, whole batch)
    cuda_sync()
    with nvtx.range('timed_prefill'):
        t0 = time.perf_counter()
        po: PrefillOutput = prefill(model, input_ids)
        cuda_sync()
        t1 = time.perf_counter()
    prefill_ms = (t1 - t0) * 1000.0

    # Decode loop (token-by-token, but each "step" generates B tokens across the batch)
    decode_step_ms: list[float] = []
    if gen_tokens > 0:
        token = po.next_token  # [B, 1]
        past = po.past_key_values

        with nvtx.range('timed_decode_loop'):
            for i in range(gen_tokens):
                cuda_sync()
                ts = time.perf_counter()
                with nvtx.range(f'decode_step_{i}'):
                    out: Any = model(input_ids=token, past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                cuda_sync()
                te = time.perf_counter()
                decode_step_ms.append((te - ts) * 1000.0)

    reserved_mb, allocated_mb = mem_mb_peaks()
    return RunMeasurement(
        prefill_ms=prefill_ms,
        decode_step_ms=decode_step_ms,
        reserved_mb_peak=reserved_mb,
        allocated_mb_peak=allocated_mb,
    )


def aggregate(
    seq_len: int, batch_size: int, gen_tokens: int, reps: list[RunMeasurement]
) -> AggregatedMetrics:
    """
    Aggregate multiple reps into p50 metrics.

    Definitions:
      - prefill_ms_p50: median prefill wall time (for the whole batch)
      - prefill_ms_per_token_p50: prefill_ms_p50 / (B * L)
      - prefill_tok_per_s_p50: (B * L) / prefill_seconds_p50

      - decode_total_ms_p50: median of sum(decode_step_ms) (for the whole batch)
      - decode_ms_per_tok_p50: median(decode_total_ms / gen_tokens)
        Interpreted as: ms per decode step where each step generates B tokens total
      - decode_tok_per_s_p50: (B * 1000) / decode_ms_per_tok_p50

      - total_tok_s: (B*L + B*gen_tokens) / (prefill_p50 + decode_total_p50)
    """
    if not reps:
        raise ValueError('reps must be non-empty')

    prefill_ms = [r.prefill_ms for r in reps]
    decode_total_ms = [sum(r.decode_step_ms) for r in reps]

    p50_prefill = median(prefill_ms)
    p50_decode_total = median(decode_total_ms)

    prompt_tokens = float(seq_len * batch_size)
    gen_total_tokens = float(gen_tokens * batch_size)

    prefill_ms_per_token_p50 = p50_prefill / prompt_tokens if prompt_tokens > 0 else 0.0
    prefill_tok_per_s_p50 = prompt_tokens / (p50_prefill / 1000.0) if p50_prefill > 0 else float('inf')

    if gen_tokens > 0:
        decode_ms_per_tok = [(dt / gen_tokens) for dt in decode_total_ms]
        p50_decode_ms_per_tok = median(decode_ms_per_tok)
        decode_tok_per_s_p50 = (
            (1000.0 * float(batch_size)) / p50_decode_ms_per_tok
            if p50_decode_ms_per_tok > 0
            else float('inf')
        )
    else:
        p50_decode_ms_per_tok = 0.0
        decode_tok_per_s_p50 = float('inf')

    total_tokens = prompt_tokens + gen_total_tokens
    total_ms = p50_prefill + p50_decode_total
    tok_s = total_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    vram_reserved = max(r.reserved_mb_peak for r in reps)
    vram_allocated = max(r.allocated_mb_peak for r in reps)

    return AggregatedMetrics(
        seq_len=seq_len,
        batch_size=batch_size,
        gen_tokens=gen_tokens,
        prefill_ms_p50=p50_prefill,
        prefill_ms_per_token_p50=prefill_ms_per_token_p50,
        prefill_tok_per_s_p50=prefill_tok_per_s_p50,
        decode_total_ms_p50=p50_decode_total,
        decode_ms_per_tok_p50=p50_decode_ms_per_tok,
        decode_tok_per_s_p50=decode_tok_per_s_p50,
        total_tok_s=tok_s,
        vram_reserved_mb=vram_reserved,
        vram_allocated_mb=vram_allocated,
    )
