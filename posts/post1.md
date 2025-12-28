
🚀 For LLM inference GPUs Want Parallelism, not longer prompts

I ran an empirical latency sensitivity study on my personal RTX 5070 GPU for LLM inference and learned something 
very clear about how GPUs respond to context length versus concurrency (batching).

💡 What hurts GPU efficiency: longer prompts for a single request

As sequence length increases (batch = 1):
- Prefill latency per token worsens: 0.27 → 0.43 ms/token (128 → 4096)
- Decode throughput drops: 38 → 28 tok/s
- VRAM pressure doubles: ~6 GB → ~12 GB

Longer context adds serial work and memory pressure which makes the GPUs less efficient.

💡 What actually works: batching

At fixed context length, increasing concurrency:
- Throughput scales from 200 -> 2800 tok/s (batch 1 → 64)
- Per-token prefill efficiency stays nearly flat (~0.18–0.19 ms/token)

GPUs remain well-utilized until saturation

🧠 Key insight

Processing independent requests in parallel is more tokens:
- Long context = one bigger dependency chain.
- Batching = many independent chains.

⚡ GPUs handle batching better than long context because batching exposes independent parallel work. Independent requests 
allow the GPU to schedule more concurrent warps, hide memory latency, and keep execution units busy. Long-context
inference adds serial, dependency-heavy work to a single request, which limits parallelism and reduces utilization.

⚙️ In production this has important implications:
✅ Batching is mandatory for efficiency, not an optimization
✅ Engines like vLLM matter when concurrency exists
❌ Naive FastAPI-style per-request serving wastes GPUs
❌ No framework can save low-concurrency workloads

🛠️ Methodology (brief)

System: 
- GPU: RTX 5070 (12 GB), 
- Model: Qwen2.5-3B 
- Precision: FP16

Process:
- Separate prefill vs decode 
- median aggregation
- controlled sweeps

Full write-up with tables, plots, and reproducible commands: [link]

#LLM #Inference #GPU #PerformanceEngineering #MLOps #Systems