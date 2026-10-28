<div align="center">

# Hybrid Reasoning Agents

**LLM Agents that Think in latent space, Act through text**

[![arXiv](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2610.18861)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-models%20%26%20data-yellow.svg)](https://huggingface.co/collections/togethercomputer/hybridagent)
[![X](https://img.shields.io/badge/X-thread-000000.svg?logo=x&logoColor=white)](https://x.com/Jiaru_Zou/status/2110400256647340476)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-post-0A66C2.svg?logo=linkedin&logoColor=white)](https://lnkd.in/p/g_3PW9PS)

[Jiaru Zou](https://scholar.google.com/citations?user=GzLTey4AAAAJ)<sup>1,2</sup>, [Yunzhe Qi](https://scholar.google.com/citations?user=Gt17_A0AAAAJ)<sup>3,4</sup>, [Sirui Chen](https://csr16.github.io/)<sup>3</sup>, [Federico Bianchi](https://federicobianchi.io/)<sup>2</sup>, [Shang Zhu](https://shang-zhu.github.io/)<sup>2</sup>, [Pan Lu](https://lupantech.github.io/)<sup>1</sup>, [Jingrui He](https://www.hejingrui.org/)<sup>3</sup>, [James Zou](https://www.james-zou.com/)<sup>1,2</sup>

<sup>1</sup>Stanford University · <sup>2</sup>Together AI · <sup>3</sup>UIUC · <sup>4</sup>Meta

</div>

<p align="center"><img src="assets/hybrid_agent_overview.png" width="100%" alt="HybridAgent architecture overview"></p>

**HybridAgent** is an LLM agent that dynamically navigates between latent reasoning and text-based interaction within a single generation process. A lightweight **HybridLink** maps last-layer hidden states back into the input-embedding space, so the agent can enter **latent mode** for fast, inspectable inner thinking and return to **text mode** for tool calls, final answers and external monitoring. The complete hybrid context of text, latent thoughts and tool observations is carried across turns through the agent's own **working memory**, and every latent thought can be rendered as readable pseudo-text.

This repository contains the method: the hybrid generation engine with Python and search tools, the **HybridSFT** and **HybridRL** post-training stages, batch inference, the **Interpretable Latent Mode** inspector, and full support for **Gemma 4 Dense (`12B`) & MoE (`26B-A4B`)** models with TensorRT and Triton W4A16 acceleration.

## ⚙️ Installation

```bash
git clone -b gemma4 https://github.com/milkshake0721/HybridAgent.git && cd HybridAgent
conda create -n hybridagent python=3.11 -y && conda activate hybridagent
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # TAVILY_API_KEY for the search tool
```

## 📁 Repository

| Directory | Contents |
|---|---|
| [`hybrid_agent/`](hybrid_agent/) | HybridLink, the model wrapper and loader, the hybrid generation engine (text/latent modes, working memory, concurrent tool calls), TensorRT/SGLang/W4A16 backends, and tools |
| [`hybrid_sft/`](hybrid_sft/) | HybridSFT: data format, objective and trainer |
| [`hybrid_rl/`](hybrid_rl/) | HybridRL: hybrid policy rollout, reward, exact rollout replay, policy update |
| [`hybrid_inference/`](hybrid_inference/) | Batch hybrid inference over a JSONL of questions |
| [`interpretable_latent_mode/`](interpretable_latent_mode/) | Latent thoughts as pseudo-text: HTML reports and terminal view |
| [`scripts/`](scripts/) | Launchers for the three stages |

## 🤗 Released models and data

Checkpoints and data load from a local directory or directly from the Hugging Face Hub.

| Artifact | Description | Hugging Face |
|---|---|---|
| **Gemma-4-26B-A4B-HybridAgent** | `google/gemma-4-26B-A4B` (26B MoE, 4B Active, BF16, ~54.7 GB VRAM) | [`milkshake721/Gemma-4-26B-A4B-HybridAgent`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent) |
| **Gemma-4-26B-A4B-HybridAgent-4bit** | 4-bit `compressed-tensors` W4A16 MoE + BF16 Attention & `HybridLink` (~19.0 GB VRAM, fits on 24GB GPU) | [`milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit) |
| **Gemma-4-12B-HybridAgent** | `google/gemma-4-12B` (12B Dense, BF16, ~24 GB VRAM) | [`milkshake721/Gemma-4-12B-HybridAgent`](https://huggingface.co/milkshake721/Gemma-4-12B-HybridAgent) |
| HybridAgent-8B-Agentic | Qwen3-8B-Base → HybridSFT → HybridRL on agentic reasoning tasks | [`togethercomputer/HybridAgent-8B-Agentic`](https://huggingface.co/togethercomputer/HybridAgent-8B-Agentic) |
| HybridAgent-14B-Agentic | Qwen3-14B-Base → HybridSFT → HybridRL on agentic reasoning tasks | [`togethercomputer/HybridAgent-14B-Agentic`](https://huggingface.co/togethercomputer/HybridAgent-14B-Agentic) |
| HybridAgent-HybridSFT-v1 / -v2 | HybridSFT training sets: hybrid trajectories (`train.jsonl`), v1 and the upgraded v2 | [`togethercomputer/HybridAgent-HybridSFT-v1`](https://huggingface.co/datasets/togethercomputer/HybridAgent-HybridSFT-v1), [`togethercomputer/HybridAgent-HybridSFT-v2`](https://huggingface.co/datasets/togethercomputer/HybridAgent-HybridSFT-v2) |
| HybridAgent-HybridRL | HybridRL training sets: deep reasoning (`math.parquet`) and agentic search (`search.parquet`) | [`togethercomputer/HybridAgent-HybridRL`](https://huggingface.co/datasets/togethercomputer/HybridAgent-HybridRL) |

*Note:* Each model directory holds the backbone in Hugging Face format together with the HybridLink weights (`adapter.pt` / `hybridlink.pt`, `adapter_config.json` / `hybridlink_config.json`).

## ⚡ Gemma 4 MoE (`26B-A4B`) — TensorRT & SGLang Acceleration Guide

Because **HybridAgent** dynamically switches between **Text Mode (`DECODE`)** and **Latent Mode (`LATENT`: continuous hidden-state feedback $e_{t+1} = \mathcal{R}(h_t)$ via `HybridLink` scaled by Gemma 4's `embed_scale = \sqrt{2816}`)** over a shared persistent KV working memory, this repository provides dedicated **TensorRT (`torch_tensorrt`)** and **SGLang (`sglang`)** backends in [`hybrid_agent/backends.py`](hybrid_agent/backends.py) and [`sglang_tensorrt_runner.py`](sglang_tensorrt_runner.py).

### 1. Install Acceleration Dependencies

```bash
pip install torch-tensorrt==2.11.0 sglang==0.5.17
```

### 2. Method A: TensorRT-Accelerated `HybridEngine` (`backend="tensorrt"`, Recommended)

Compiles `HybridLink` (`adapter.pt` / `hybridlink.pt`) into a native **NVIDIA TensorRT Engine** (`torch_tensorrt.compile`), patches all 30 BF16 MoE layers (`Gemma4TextExperts`) with a **Zero-Sync Batched Fast Path** (eliminating CPU-GPU `.nonzero()` syncs per step), and fuses the 4-bit (`Gemma4TextExperts4Bit`) single-token W4A16 decode kernel via Triton/Inductor:

```bash
# Run 4-bit W4A16 checkpoint (~19.0 GB VRAM) with TensorRT backend
python3 sglang_tensorrt_runner.py \
    --backend tensorrt \
    --model milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit \
    --question "A rectangle has a perimeter of 30 cm and an area of 54 cm^2. What are the lengths of its sides?"

# Run Full BF16 checkpoint (~54.7 GB VRAM) with TensorRT backend
python3 sglang_tensorrt_runner.py \
    --backend tensorrt \
    --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
    --question "If 3x + 7 = 22, what is the value of 5x - 2?"
```

Or via the Python API:

```python
import torch
from hybrid_agent import HybridEngine, SYSTEM_PROMPT, load_hybrid_agent

model, tokenizer = load_hybrid_agent(
    "milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit",  # or "milkshake721/Gemma-4-26B-A4B-HybridAgent"
    dtype=torch.bfloat16,
    device="cuda",
    backend="tensorrt",
)
engine = HybridEngine(model, tokenizer, max_new_tokens=1024, max_latent_steps=64, temperature=0.0)
traj = engine.run("If 3x + 7 = 22, what is the value of 5x - 2?", system_prompt=SYSTEM_PROMPT)
print("Final Answer:", traj.final_answer)
print(traj.output_rendered)
```

### 3. Method B: SGLang Hybrid Engine & Server (`SGLangHybridEngine` / `sglang.launch_server`)

SGLang (`>=0.5.17`) natively supports both Gemma 4 MoE checkpoint layouts in `sglang/srt/models/gemma4_causal.py`:
- **4-bit `compressed-tensors` per-expert layout** (`milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit`)
- **BF16 fused `experts.gate_up_proj` layout** (`milkshake721/Gemma-4-26B-A4B-HybridAgent`)

```bash
# 1) Run via SGLangHybridEngine (couples SGLang RadixAttention with TensorRT HybridLink & Python tool executor)
python3 sglang_tensorrt_runner.py \
    --backend sglang \
    --model milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit \
    --question "If 3x + 7 = 22, what is the value of 5x - 2?"

# 2) Or launch a standalone SGLang OpenAI-compatible Server:
# For 4-bit W4A16 checkpoint:
python3 -m sglang.launch_server \
    --model-path milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit \
    --quantization compressed-tensors \
    --dtype bfloat16 \
    --port 30000 \
    --mem-fraction-static 0.80 \
    --trust-remote-code

# For BF16 checkpoint:
python3 -m sglang.launch_server \
    --model-path /path/to/Gemma-4-26B-A4B-HybridAgent \
    --dtype bfloat16 \
    --port 30000 \
    --mem-fraction-static 0.80 \
    --trust-remote-code
```

### 4. Benchmark Results (RTX PRO 6000 Blackwell)

See [`BENCHMARK_AND_OPTIMIZATION_NOTES.md`](BENCHMARK_AND_OPTIMIZATION_NOTES.md) for the kernel fusion architecture and per-question breakdown.

| Model / Backend | Allocated VRAM | Prefill (180 tok) | Single-Token Decode | 10-Q Total Time | Avg Time / Q | 10-Q Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Gemma-4-26B-A4B-HybridAgent` **BF16** (`backend="tensorrt"`) | `54.73 GB` | `313.03 ms` | `26.01 ms` (`38.5 tok/s`) | `106.80s` | `10.68s` | **`10/10 (100%)`** |
| 🏆 **`Gemma-4-26B-A4B-HybridAgent-4bit`** (`backend="tensorrt"`, Fused Triton W4A16 + `F.rms_norm` + Compiled FFN) | **`19.00 GB`** | **`187.51 ms`** | **`16.62 ms` (`60.2 tok/s`)** | **`83.65s`** | **`8.36s`** | **`10/10 (100%)`** |

```bash
# Single-question verification suite (2 questions)
python3 test_gemma4_26b_hybrid.py milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit --backend tensorrt
python3 test_gemma4_26b_hybrid.py milkshake721/Gemma-4-26B-A4B-HybridAgent --backend tensorrt

# 10-question math/logic benchmark evaluation
python3 eval_gemma4_26b_hybrid.py milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit eval_results_4bit.json tensorrt
```

## 🚀 Quick start

### Batch Hybrid Inference

```bash
python -m hybrid_inference.run_batch --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
    --input examples/questions.jsonl --output outputs/results.jsonl --num_samples 4
bash scripts/run_batch_inference.sh 8 --model togethercomputer/HybridAgent-8B-Agentic \
    --input questions.jsonl --output outputs/results.jsonl        # one process per GPU
```

One record per question: the hybrid trajectory, the extracted answer, the pseudo-text readout of every latent span and the tool-call statistics. See [`hybrid_inference/`](hybrid_inference/).

### Interpretable latent mode

```bash
python -m interpretable_latent_mode --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
    --question "How many trailing zeros does 100! have?" --output report.html
```

Renders the latent thoughts as pseudo-text inline with the text reasoning, tool calls and observations. HybridAgent-14B-Agentic on the first question of `examples/questions.jsonl` (terminal view, text segments omitted):

```text
QUESTION  At the speed of sound in air at 20 degrees Celsius, how many seconds does sound take
          to travel 10 kilometers? Round to the nearest second.

LATENT    [span 1 · 42 steps · readability 0.84]
          Wait, I me the the speed of speed in air at 20° is about 343 meters per second. But I
          should verify that. Let me do a search search to be sure.
SEARCH    speed of sound in air at 20 degrees celsius meters per second
RESULT    Page 1: The speed of sound in dry air at 20 °C is 343 meters per second. ...

LATENT    [span 2 · 22 steps · readability 0.88]
          Now, I need to calculate the time it takes for speed to travel 10 km at this speed.
LATENT    [span 3 · 21 steps · readability 0.96]
          So, time = 10,000 meters / 343 meters per second.
PYTHON    ... print(f"Rounded to the nearest second: {rounded_time} seconds")
RESULT    Time taken: 29.154518950437318 seconds
          Rounded to the nearest second: 29 seconds

LATENT    [span 7 · 10 steps · readability 0.92]
          So, the answer is 29 seconds.
ANSWER    \boxed{29}
```

See [`interpretable_latent_mode/`](interpretable_latent_mode/) for the full trace and the HTML report.

### HybridSFT

```bash
MODEL=Qwen/Qwen3-8B-Base DATA=togethercomputer/HybridAgent-HybridSFT-v2 OUT=outputs/hybrid_sft \
    bash scripts/run_hybrid_sft.sh all
```

Train HybridSFT on the curated hybrid trajectories (format activation, HybridLink activation and self-conditioned training). Recommended hardware: 8 or 16 GPUs. See [`hybrid_sft/`](hybrid_sft/).

### HybridRL

```bash
TASK=search MODEL=outputs/hybrid_sft/self_conditioned DATA=hf://togethercomputer/HybridAgent-HybridRL/search.parquet \
    NNODES=4 NODE_RANK=<node> MASTER_ADDR=<rank-0 host> bash scripts/run_hybrid_rl.sh
```

Train HybridRL on the agentic search tasks. Recommended hardware: 32 GPUs (4 nodes × 8), the launcher default. See [`hybrid_rl/`](hybrid_rl/).

## 📚 Citation

```bibtex
@misc{zou2026hybridagent,
      title={Hybrid Reasoning Agents}, 
      author={Jiaru Zou and Yunzhe Qi and Sirui Chen and Federico Bianchi and Shang Zhu and Pan Lu and Jingrui He and James Zou},
      year={2026},
      eprint={2610.18861},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2610.18861}, 
}
```
