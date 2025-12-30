# src/llm_latency_anatomy/sweep.py
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .collapse import detect as detect_collapse
from .metrics import AggregatedMetrics, aggregate, measure_one


@dataclass(frozen=True)
class RunStats:
    """
    Structured summary for a single sweep point (one seq_len x batch_size).
    """

    seq_len: int
    batch_size: int
    gen_tokens: int

    prefill_ms_p50: float
    prefill_ms_per_token_p50: float
    prefill_tok_per_s_p50: float

    decode_total_ms_p50: float
    decode_ms_per_tok_p50: float
    decode_tok_per_s_p50: float

    total_tok_s: float
    vram_reserved_mb: float
    vram_allocated_mb: float
    collapse_flag: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype_from_str(s: str) -> torch.dtype:
    s2 = s.strip().lower()
    if s2 == 'fp16':
        return torch.float16
    if s2 == 'bf16':
        return torch.bfloat16
    raise ValueError(f'Unknown dtype: {s!r}. Supported: fp16, bf16')


def _parse_int_list(csv_s: str) -> list[int]:
    items = [x.strip() for x in csv_s.split(',') if x.strip()]
    vals = [int(x) for x in items]
    if not vals:
        raise ValueError('List must be non-empty.')
    if any(v <= 0 for v in vals):
        raise ValueError('All values must be positive integers.')
    return vals


def _make_prompt_tokens(tokenizer: Any, target_len: int, batch_size: int) -> torch.Tensor:
    """
    Build B distinct synthetic prompts of exactly `target_len` tokens.

    Returns:
      - CPU LongTensor [B, L], caller moves it to CUDA.
    """
    base = 'The quick brown fox jumps over the lazy dog. '
    rows: list[torch.Tensor] = []

    for i in range(batch_size):
        text = f'Request {i}. ' + (base * 2048)  # overshoot, then trim by tokens
        ids = tokenizer(text, return_tensors='pt').input_ids[0]
        if ids.numel() < target_len:
            reps = (target_len + ids.numel() - 1) // ids.numel()
            ids = ids.repeat(reps)
        rows.append(ids[:target_len])

    return torch.stack(rows, dim=0)


def _nvtx_push(msg: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


# ----------------------------
# Plotting: combined figure(s)
# ----------------------------


def _read_results_csv_for_plots(path: str) -> list[dict[str, Any]]:
    """
    Read the sweep CSV and return rows with numeric fields converted for plotting.
    """
    rows: list[dict[str, Any]] = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            row['seq_len'] = int(row['seq_len'])
            row['batch_size'] = int(row['batch_size'])

            row['prefill_ms_per_token_p50'] = float(row['prefill_ms_per_token_p50'])
            row['decode_ms_per_tok_p50'] = float(row['decode_ms_per_tok_p50'])
            row['decode_tok_per_s_p50'] = float(row['decode_tok_per_s_p50'])
            row['total_tok_s'] = float(row['total_tok_s'])
            row['vram_reserved_mb'] = float(row['vram_reserved_mb'])
            rows.append(row)
    return rows


def _write_combined_batch_sweep_plots(out_csv: str, images_dir: str) -> None:
    """
    One image per seq_len that has multiple batch points.

    3 subplots (shared X: batch size, log2):
      1) prefill_ms_per_token_p50
      2) decode_tok_per_s_p50
      3) total_tok_s
    """
    rows = _read_results_csv_for_plots(out_csv)
    if not rows:
        return

    by_seq: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_seq.setdefault(r['seq_len'], []).append(r)

    out_dir = Path(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(out_csv).stem

    for L, rs in by_seq.items():
        batches = sorted({int(r['batch_size']) for r in rs})
        if len(batches) < 2:
            continue

        rs_sorted = sorted(rs, key=lambda x: int(x['batch_size']))
        x = [int(r['batch_size']) for r in rs_sorted]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

        axes[0].plot(
            x, [r['prefill_ms_per_token_p50'] for r in rs_sorted], marker='o', label='prefill_ms/tok'
        )
        axes[0].set_ylabel('Prefill (ms/tok)')
        axes[0].set_title(f'Batch sweep (seq_len={L})')
        axes[0].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[0].legend()

        axes[1].plot(x, [r['decode_tok_per_s_p50'] for r in rs_sorted], marker='o', label='decode_tok/s')
        axes[1].set_ylabel('Decode (tok/s)')
        axes[1].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[1].legend()

        axes[2].plot(x, [r['total_tok_s'] for r in rs_sorted], marker='o', label='total_tok/s')
        axes[2].set_ylabel('Total (tok/s)')
        axes[2].set_xlabel('Batch size')
        axes[2].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[2].legend()

        axes[2].set_xscale('log', base=2)

        out_path = out_dir / f'{stem}__seq_{L}__batch_sweep__combined.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)


def _write_combined_seq_sweep_plots(out_csv: str, images_dir: str) -> None:
    """
    One image per batch_size that has multiple seq_len points.

    3 subplots (shared X: seq_len):
      1) prefill_ms_per_token_p50
      2) decode_ms_per_tok_p50   <-- key for seq scaling
      3) total_tok_s
    """
    rows = _read_results_csv_for_plots(out_csv)
    if not rows:
        return

    by_bs: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        by_bs.setdefault(r['batch_size'], []).append(r)

    out_dir = Path(images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(out_csv).stem

    for B, rs in by_bs.items():
        seqs = sorted({int(r['seq_len']) for r in rs})
        if len(seqs) < 2:
            continue

        rs_sorted = sorted(rs, key=lambda x: int(x['seq_len']))
        x = [int(r['seq_len']) for r in rs_sorted]

        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))

        axes[0].plot(
            x, [r['prefill_ms_per_token_p50'] for r in rs_sorted], marker='o', label='prefill_ms/tok'
        )
        axes[0].set_ylabel('Prefill (ms/tok)')
        axes[0].set_title(f'Sequence sweep (batch={B})')
        axes[0].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[0].legend()

        axes[1].plot(x, [r['decode_ms_per_tok_p50'] for r in rs_sorted], marker='o', label='decode_ms/tok')
        axes[1].set_ylabel('Decode (ms/tok)')
        axes[1].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[1].legend()

        axes[2].plot(x, [r['total_tok_s'] for r in rs_sorted], marker='o', label='total_tok/s')
        axes[2].set_ylabel('Total (tok/s)')
        axes[2].set_xlabel('Sequence length (tokens)')
        axes[2].grid(True, which='both', linestyle='--', alpha=0.4)
        axes[2].legend()

        out_path = out_dir / f'{stem}__bs_{B}__seq_sweep__combined.png'
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close(fig)


def _write_all_combined_plots(out_csv: str, images_dir: str = 'results/sweeps/nsys') -> None:
    _write_combined_batch_sweep_plots(out_csv, images_dir=images_dir)
    _write_combined_seq_sweep_plots(out_csv, images_dir=images_dir)


def run_sweep(
    model_id: str,
    dtype: str,
    device: str,
    batch_size: int,
    batch_sizes_csv: str | None,
    gen_tokens: int,
    warmup: int,
    reps: int,
    seq_csv: str,
    single_seq: int | None,
    out_csv: str,
    seed: int,
    collapse_factor: float,
) -> None:
    """
    Sweep over (seq_len x batch_size) grid and measure prefill + decode latency.
    """
    assert device == 'cuda', 'This tool currently supports CUDA only.'
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. This tool targets an RTX GPU run.')
    if gen_tokens < 0:
        raise ValueError('--gen-tokens must be >= 0.')
    if warmup < 0 or reps <= 0:
        raise ValueError('--warmup must be >=0 and --reps must be >0.')

    _set_seed(seed)
    torch.set_float32_matmul_precision('high')
    dt = _dtype_from_str(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dt,
        device_map='cuda',
        low_cpu_mem_usage=True,
    ).eval()

    seq_points = _parse_int_list(seq_csv)
    if single_seq is not None:
        seq_points = [single_seq]

    if batch_sizes_csv and batch_sizes_csv.strip():
        batch_sizes = _parse_int_list(batch_sizes_csv)
    else:
        if batch_size <= 0:
            raise ValueError('--batch-size must be positive.')
        batch_sizes = [batch_size]

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(
            [
                'seq_len',
                'batch_size',
                'gen_tokens',
                'prefill_ms_p50',
                'prefill_ms_per_token_p50',
                'prefill_tok_per_s_p50',
                'decode_total_ms_p50',
                'decode_ms_per_tok_p50',
                'decode_tok_per_s_p50',
                'total_tok_s',
                'vram_reserved_mb',
                'vram_allocated_mb',
                'collapse_flag',
            ]
        )

        print(
            'seq_len | batch | prefill_p50(ms) | prefill_ms/tok | prefill_tok/s | '
            'decode_p50(ms/tok) | decode_tok/s | total_tok/s | VRAM_reserved(MB) | flag'
        )
        print('-' * 150)

        # Run collapse detection within each batch-size group
        for B in batch_sizes:
            prev_metrics: AggregatedMetrics | None = None
            collapse_point: int | None = None

            for L in seq_points:
                torch.cuda.reset_peak_memory_stats()

                input_ids = _make_prompt_tokens(tokenizer, L, B).to('cuda')

                with torch.inference_mode():
                    for _ in range(warmup):
                        _ = measure_one(model, input_ids, gen_tokens)

                    _nvtx_push(f'nsys_measured_seq_{L}_bs_{B}')
                    try:
                        rep_measurements = [measure_one(model, input_ids, gen_tokens) for _ in range(reps)]
                    finally:
                        _nvtx_pop()

                cur = aggregate(seq_len=L, batch_size=B, gen_tokens=gen_tokens, reps=rep_measurements)

                if prev_metrics is None:
                    flag = ''
                else:
                    collapse = detect_collapse(prev_metrics, cur, collapse_factor=collapse_factor)
                    flag = collapse.flag

                if flag and collapse_point is None:
                    collapse_point = L

                w.writerow(
                    [
                        L,
                        B,
                        gen_tokens,
                        f'{cur.prefill_ms_p50:.3f}',
                        f'{cur.prefill_ms_per_token_p50:.6f}',
                        f'{cur.prefill_tok_per_s_p50:.3f}',
                        f'{cur.decode_total_ms_p50:.3f}',
                        f'{cur.decode_ms_per_tok_p50:.6f}',
                        f'{cur.decode_tok_per_s_p50:.3f}',
                        f'{cur.total_tok_s:.3f}',
                        f'{cur.vram_reserved_mb:.1f}',
                        f'{cur.vram_allocated_mb:.1f}',
                        flag,
                    ]
                )

                print(
                    f'{L:6d} | {B:5d} | {cur.prefill_ms_p50:14.2f} | {cur.prefill_ms_per_token_p50:13.6f} | {cur.prefill_tok_per_s_p50:12.1f} | '
                    f'{cur.decode_ms_per_tok_p50:16.3f} | {cur.decode_tok_per_s_p50:11.2f} | {cur.total_tok_s:9.2f} | '
                    f'{cur.vram_reserved_mb:16.1f} | {flag}'
                )

                prev_metrics = cur

            if collapse_point is not None:
                print(f'\n>>> Batch {B}: first collapse point at seq_len = {collapse_point}\n')

    print(f'\nWrote results to: {out_csv}')

    try:
        _write_all_combined_plots(out_csv, images_dir='results/sweeps/nsys')
        print('Wrote combined plots to: results/sweeps/nsys')
    except Exception as e:
        print(f'[WARN] Plot generation failed: {e}')
