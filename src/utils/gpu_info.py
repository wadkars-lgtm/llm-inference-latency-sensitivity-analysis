import json
import re
import shutil
import subprocess
from pathlib import Path


def find_nvidia_smi() -> str:
    p = shutil.which('nvidia-smi')
    if p:
        return p

    win_default = Path(r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    if win_default.exists():
        return str(win_default)

    raise FileNotFoundError(
        'nvidia-smi not found on PATH and not found at '
        r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
    )


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def parse_q_minimal(q_text: str) -> dict:
    def grab(pattern: str):
        m = re.search(pattern, q_text, re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        # identity
        'name': grab(r'^\s*Product Name\s*:\s*(.+)$'),
        'architecture': grab(r'^\s*Product Architecture\s*:\s*(.+)$'),
        'driver_version': grab(r'^\s*Driver Version\s*:\s*(.+)$'),
        'cuda_version': grab(r'^\s*CUDA Version\s*:\s*(.+)$'),
        'vbios': grab(r'^\s*VBIOS Version\s*:\s*(.+)$'),
        'pci_bus_id': grab(r'^\s*Bus Id\s*:\s*(.+)$'),
        # memory
        'fb_memory_total_mib': grab(r'^\s*Total\s*:\s*([0-9]+)\s*MiB$'),
        'fb_memory_used_mib': grab(r'^\s*Used\s*:\s*([0-9]+)\s*MiB$'),
        'fb_memory_free_mib': grab(r'^\s*Free\s*:\s*([0-9]+)\s*MiB$'),
        # clocks
        'graphics_clock_mhz': grab(r'^\s*Graphics\s*:\s*([0-9]+)\s*MHz$'),
        'memory_clock_mhz': grab(r'^\s*Memory\s*:\s*([0-9]+)\s*MHz$'),
        'max_graphics_clock_mhz': grab(r'^\s*Max Clocks.*\n\s*Graphics\s*:\s*([0-9]+)\s*MHz$'),
        # state
        'pstate': grab(r'^\s*Performance State\s*:\s*(P[0-9]+)$'),
        'temperature_gpu_c': grab(r'^\s*GPU Current Temp\s*:\s*([0-9]+)\s*C$'),
    }


def gpu_info_from_smi() -> dict:
    smi = find_nvidia_smi()

    # Expanded but still conservative field list
    fields = [
        # identity
        'name',
        'driver_version',
        'pci.bus_id',
        # memory
        'memory.total',
        'memory.used',
        'memory.free',
        # clocks
        'clocks.current.graphics',
        'clocks.current.memory',
        'clocks.max.graphics',
        'clocks.max.memory',
        # power / thermals
        'temperature.gpu',
        'power.draw',
        'power.limit',
        # utilization / state
        'utilization.gpu',
        'utilization.memory',
        'pstate',
    ]

    try:
        out = run([smi, f"--query-gpu={','.join(fields)}", '--format=csv,noheader,nounits'])

        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        gpus = []

        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            gpus.append(dict(zip(fields, parts)))

        return {
            'method': 'query',
            'gpus': gpus,
        }

    except subprocess.CalledProcessError:
        # Fallback: full -q parse
        q = run([smi, '-q'])
        return {
            'method': 'q_fallback',
            'gpus': [parse_q_minimal(q)],
            'raw_q': q,
        }


if __name__ == '__main__':
    print(json.dumps(gpu_info_from_smi(), indent=2))
