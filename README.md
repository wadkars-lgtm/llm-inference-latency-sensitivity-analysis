# LLM Latency sensitivity analysis (on RTX 5070)


A systems-level, kernel-attributed analysis of LLM latency sensitivity under long-context inference on RTX 5070 GPUs.

> Note: LLM tools were used to accelerate benchmark scaffolding. The value of this work lies in experimental design, 
> measurement, and interpretation rather than manual code construction.

## Development

* Clone this repository
* Requirements:
  * `angreal`
* `pip install angreal && angreal setup`


This project was generated using the [angreal python template](https://github.com/angreal/python) template.

## Install

#### 1. First install
```shell
 pip install -e .
```


#### 2. Install the appropriate version of Pytorch

Find what versions of CUDA drivers you have with
```shell

#[notice] To update, run: python.exe -m pip install --upgrade pip
(.venv) PS nvidia-smi
Fri Dec 19 13:38:44 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 577.05                 Driver Version: 577.05         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5070 ...  WDDM  |   00000000:02:00.0  On |                  N/A |
| N/A   38C    P8              3W /   95W |     298MiB /  12227MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+

```

This requires torch version below:

```shell
 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda runtime', torch.version.cuda); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

##Output

torch 2.9.1+cu128
cuda available True
cuda runtime 12.8
gpu NVIDIA GeForce RTX 5070 Ti Laptop GPU

```

## Let's go

```shell
huggingface-cli login
#Login to HF
```
#### Sequence length sweep (batch size = 1):

```shell
latency-sensitivity sweep `
  --model Qwen/Qwen2.5-3B-Instruct `
  --dtype fp16 `
  --device cuda `
  --seq 128,256,512,1024,2048,3072,4096 `
  --batch-sizes 1 `
  --gen-tokens 32 `
  --warmup 3 `
  --reps 7 `
  --out results/sweeps/qwen25_3b_fp16_grid_fixed_batch.csv
```

#### Batching sweep (fixed sequence length, increasing batch size):


```shell
latency-sensitivity sweep `
  --model Qwen/Qwen2.5-3B-Instruct `
  --dtype fp16 `
  --device cuda `
  --seq 128 `
  --batch-sizes 1,2,4,8,16,32,64 `
  --gen-tokens 32 `
  --warmup 3 `
  --reps 7 `
  --collapse-factor 2 `
  --out results/sweeps/qwen25_3b_fp16_grid.csv
```

## [Full Article](LATENCY_SENSITIVITY_ANALYSIS_ON_RTX_5070.md)

## NSight Systems Profiling

### 1. Short context + high concurrency

seq_len = 128, batch = 32

This represents the efficient, parallel regime.
```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.1\target-windows-x64\nsys.exe" profile `
  --force-overwrite=true `
  --output=results/nsys/qwen_s128_b32 `
  --sample=none `
  python -m llm_latency_anatomy.__main__ sweep `
    --model Qwen/Qwen2.5-3B-Instruct `
    --dtype fp16 `
    --device cuda `
    --seq 128 `
    --batch-sizes 32 `
    --gen-tokens 32 `
    --warmup 3 `
    --reps 7 `
    --collapse-factor 2 `
    --out results/sweeps/nsight/qwen_s128_b32.csv
 ```


```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.1\host-windows-x64\nsys-ui.exe" `
  "results\nsys\qwen_s128_b32.nsys-rep"
```
### 2. Long context + single request

seq_len = 4096, batch = 1

This represents the memory- and dependency-heavy regime.

```powershell
  & "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.1\target-windows-x64\nsys.exe" profile `
  --force-overwrite=true `
  --output=results/nsys/qwen_s4096_b1 `
  --sample=none `
  python -m llm_latency_anatomy.__main__ sweep `
  --model Qwen/Qwen2.5-3B-Instruct `
  --dtype fp16 `
  --device cuda `
  --seq 4096 `
  --batch-sizes 1 `
  --gen-tokens 32 `
  --warmup 3 `
  --reps 7 `
  --collapse-factor 2 `
  --out results/sweeps/nsight/qwen_s4096_b1.csv

```

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.1\host-windows-x64\nsys-ui.exe" `
  "results\nsys\qwen_s4096_b1.nsys-rep"
```

## NCompute profiling

### 1. For batched case (seq=128, batch=32):

```powershell
ncu --set=full `
    --target-processes all `
    --kernel-name regex:.*attention.* `
    --launch-skip 3 `
    --launch-count 1 `
    --export results/ncu/qwen_s128_b32 `
    python -m llm_latency_anatomy.__main__ sweep `
      --model Qwen/Qwen2.5-3B-Instruct `
      --dtype fp16 `
      --device cuda `
      --seq 128 `
      --batch-sizes 32 `
      --gen-tokens 32 `
      --warmup 3 `
      --reps 1 `
      --out results/sweeps/ncompute/qwen_s128_b32.csv
```

### 2.And for long-context case (seq=4096, batch=1):

```powershell
ncu --set=full `
    --target-processes all `
    --kernel-name regex:.*attention.* `
    --launch-skip 3 `
    --launch-count 1 `
    --export results/ncu/qwen_s4096_b1 `
    python -m llm_latency_anatomy.__main__ sweep `
      --model Qwen/Qwen2.5-3B-Instruct `
      --dtype fp16 `
      --device cuda `
      --seq 4096 `
      --batch-sizes 1 `
      --gen-tokens 32 `
      --warmup 3 `
      --reps 1 `
      --out --out results/sweeps/ncompute/qwen_s4096_b1.csv
```