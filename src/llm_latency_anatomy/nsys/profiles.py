from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NsysProfileConfig:
    """
    Configuration for generating an Nsight Systems profile command.
    """

    out_dir: Path
    name: str
    python_module: str = 'llm_latency_anatomy.__main__'
    capture_range: str = 'nvtx'
    sample: str = 'none'


def build_nsys_command(cfg: NsysProfileConfig, python_args: list[str]) -> list[str]:
    """
    Build an `nsys profile ... python -m <module> ...` command.

    Returns:
        list[str] command suitable for printing or subprocess execution.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / cfg.name

    return [
        'nsys',
        'profile',
        '--force-overwrite=true',
        f'--output={out_path}',
        f'--capture-range={cfg.capture_range}',
        '--capture-range-end=stop',
        f'--sample={cfg.sample}',
        'python',
        '-m',
        cfg.python_module,
        *python_args,
    ]
