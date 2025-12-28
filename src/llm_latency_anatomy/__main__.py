from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .sweep import run_sweep


def _add_sweep_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--model',
        required=True,
        help='Hugging Face model name or local path',
    )
    parser.add_argument(
        '--dtype',
        default='fp16',
        choices=['fp16', 'bf16'],
        help='Model dtype',
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='Execution device (currently only cuda is supported)',
    )
    parser.add_argument(
        '--batch-sizes',
        required=True,
        help='Comma-separated list of batch sizes (e.g. "1,2,4,8,16")',
    )
    parser.add_argument(
        '--seq',
        required=True,
        help='Comma-separated list of sequence lengths (e.g. "256,512,1024")',
    )
    parser.add_argument(
        '--gen-tokens',
        type=int,
        default=128,
        help='Number of decode tokens per run (per sequence)',
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Warmup runs per (seq_len, batch_size)',
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=5,
        help='Measured repetitions per (seq_len, batch_size)',
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output CSV path',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed',
    )
    parser.add_argument(
        '--collapse-factor',
        type=float,
        default=8.0,
        help='Factor threshold for declaring latency collapse (within same batch size)',
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog='latency-collapse',
        description='Latency behavior analysis for long-context LLM inference',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    sweep_parser = subparsers.add_parser(
        'sweep',
        help='Run (seq_len × batch_size) latency grid sweep',
    )
    _add_sweep_args(sweep_parser)

    args = parser.parse_args(argv)

    if args.command == 'sweep':
        run_sweep(
            model_id=args.model,
            dtype=args.dtype,
            device=args.device,
            batch_size=0,  # unused, kept for signature compatibility
            batch_sizes_csv=args.batch_sizes,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            reps=args.reps,
            seq_csv=args.seq,
            single_seq=None,
            out_csv=args.out,
            seed=args.seed,
            collapse_factor=args.collapse_factor,
        )
    else:
        raise RuntimeError(f'Unknown command: {args.command!r}')


if __name__ == '__main__':
    main(sys.argv[1:])
