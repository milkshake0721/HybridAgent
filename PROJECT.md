# Project: HybridAgent High-Concurrency TensorRT Optimization & G4 GPU 5-Benchmark Suite

## Architecture

### Overview
`HybridAgent` implements a hybrid latent-and-text reasoning architecture (`HybridAgentModel` = `Gemma4ForConditionalGeneration` backbone + `HybridLink` continuous hidden-state recurrent adapter + persistent KV working memory with `<python>` and `<search>` tool execution) for `milkshake721/Gemma-4-26B-A4B-HybridAgent` (BF16, 54.73 GB VRAM) and `milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit` (W4A16 `compressed-tensors` pack-quantized MoE, 19.00 GB VRAM), targeting Google Cloud **G4 GPU** (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7 VRAM, CUDA 13.0, PyTorch 2.11).

```text
User CLI / Benchmark Suite (eval_hf_benchmarks.py, eval_gemma4_26b_hybrid.py, run_batch.py, sglang_tensorrt_runner.py)
  │
  ├─► [Module 3: Benchmark & Runner Layer] (hybrid_agent/benchmarks/*, eval_hf_benchmarks.py, run_batch.py)
  │     ├─► Dataset Loaders & Normalizers:
  │     │     1. MMLU Pro (TIGER-Lab/MMLU-Pro, 10-choice A-J, target 82.6%)
  │     │     2. AIME 2026 No Tools (MathArena/aime_2026, integer 000-999, strict no-tools, target 88.3%)
  │     │     3. GPQA Diamond (Idavidrein/gpqa / fingertap/GPQA-Diamond, 4-choice A-D, target 82.3%)
  │     │     4. LiveCodeBench v6 (livecodebench/code_generation_lite release_v6, pass@1 sandbox, target 77.1%)
  │     │     5. MMMU Pro Multimodal (MMMU/MMMU_Pro standard-10 & vision, PIL images + text, target 73.8%)
  │     ├─► G4 GPU Concurrency Auto-Tuner & Continuous Batch Scheduler
  │     └─► Graders, Telemetry (tok/s, q/s, VRAM, latency) & Target Comparison Reporter
  │
  ├─► [Module 2: Multi-Question Engine & Multimodal Layer] (hybrid_agent/engine.py, model.py, sampling.py, readout.py)
  │     ├─► load_hybrid_agent(model_id, backend="tensorrt") + AutoProcessor (Gemma4Processor fallback)
  │     ├─► HybridEngine.run_batch(questions, images=..., batch_size=..., max_concurrency=...)
  │     └─► HybridGenerationEngine (Multi-Question Lockstep + Continuous Slot Compaction/Refill):
  │           ├─► Left-padded multi-prompt prefill + multimodal pixel_values vision encoding
  │           ├─► Vectorized DECODE sampling (zero per-row .item() syncs, pre-allocated attn_mask, gated _row_tail)
  │           ├─► Batched LATENT recurrence (lazy BF16 _BANK_CACHE readout, GPU-vectorized exit detection)
  │           └─► Multi-token chunked FEED prefill for tool <result> observations
  │
  └─► [Module 1: TensorRT & Zero-Sync MoE Backend Layer] (hybrid_agent/backends.py, quantization.py)
        ├─► Dynamic-Batch TensorRT HybridLink Engine (_TensorRTHybridLinkWrapper, min=1, opt=32, max=256)
        ├─► Zero-Sync BF16 MoE Fast Path (_fused_bf16_moe_kernel / grouped MoE for arbitrary B=1..256+, no .nonzero() syncs)
        ├─► Compiled Post-Attention FFN+Router+RMSNorm Graph for BOTH BF16 and 4-Bit (patch_compiled_ffn_bf16 / 4bit)
        ├─► Expert-Grouped / L2-Locality Optimized Triton W4A16 MoE Kernel for high-concurrency T > 8
        └─► SGLangHybridEngine multi-request concurrency (run_batch) & non-mutating latent recurrence fix
```

---

## Feature Inventory

Every feature discovered during the Phase 0 Survey (merged from `explorer_survey_1`, `explorer_survey_2`, and `explorer_survey_3`) is listed below with its assigned milestone. No feature is left unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| **F1** | **Dynamic-Batch TensorRT `HybridLink` & Graph Acceleration** | Replace static `shape=(1, hidden_size)` and `h_2d.shape[0] == 1` guard in `compile_hybridlink_tensorrt` and `_TensorRTHybridLinkWrapper` (`hybrid_agent/backends.py:306-383`) with dynamic batch profiles (`min_shape=(1, H)`, `opt_shape=(32, H)`, `max_shape=(256, H)`), chunking for `N > max_batch`, dynamic `torch.compile(..., dynamic=True)` fallback, and stream/graph concurrency tuning in `enable_tensorrt_acceleration`. | **M1** | `survey_1` (`OPT-1`), `survey_2`, `survey_3` |
| **F2** | **High-Concurrency Zero-Sync BF16 MoE & Compiled BF16 FFN** | Remove the `n_tokens <= 8` ceiling and `.nonzero()` CPU-GPU sync loop in `_bf16_experts_fast_forward` (`hybrid_agent/backends.py:209-283`) and eliminate `B*K` full expert matrix HBM duplication by implementing a zero-sync batched/grouped BF16 MoE fast path (Triton BF16 MoE kernel + grouped dispatch scaling to `B=1..256+`). Add `patch_compiled_ffn_bf16` (alongside `patch_compiled_ffn_4bit`) so BF16 `milkshake721/Gemma-4-26B-A4B-HybridAgent` fuses post-attention `RMSNorm + Shared MLP + Router + MoE + 6 RMSNorms`. | **M1** | `survey_1` (`OPT-2`), `survey_2`, `survey_3` |
| **F3** | **High-Concurrency Triton W4A16 MoE & `SGLangHybridEngine` Fix** | Optimize `fused_triton_4bit_moe_forward` / `_fused_w4a16_moe_kernel` (`hybrid_agent/quantization.py:25-130`) for high concurrency (`T > 8`) via expert-sorted/grouped token ordering for L2 weight locality while retaining low-latency small-`T` execution. Fix `SGLangHybridEngine.run` (`backends.py:765-797`) so it does not permanently mutate `self.latent_mode = "text_only"` and add concurrent `SGLangHybridEngine.run_batch`. | **M1** | `survey_1` (`OPT-3`), `survey_2`, `survey_3` |
| **F4** | **Multi-Question Batched & Concurrent `HybridGenerationEngine` / `HybridEngine`** | Extend `HybridGenerationEngine` (`hybrid_agent/engine.py:70-248`) to accept `questions: str | Sequence[str]` (multiple distinct prompts of different lengths) with left-padded batched prefill, per-row `prompt_len`, initial `positions`, and `attn_mask`. Add `HybridEngine.run_batch(questions, *, images=None, batch_size=..., max_concurrency=...)` with dynamic queue scheduling / continuous slot refilling. | **M2** | `survey_1` (`OPT-4`), `survey_2`, `survey_3` |
| **F5** | **Vectorized Step Execution, Multi-Token `FEED` Prefill & Lazy BF16 Readout** | Vectorize `sample_rows` (`sampling.py:100-110`) for greedy (`argmax` + single `.tolist()`) and stochastic (`batched multinomial` + single `.tolist()`); eliminate per-row `readout_ids[i].item()` syncs (`engine.py:414`); pre-allocate `self.attn_mask` buffer (`engine.py:266`); gate `_row_tail` `tokenizer.decode` (`engine.py:324, 520`) on tag-closing characters (`>`); implement multi-token batched/chunked `FEED` prefill for tool `<result>` observations (`engine.py:441-457`); convert `_BANK_CACHE` (`readout.py:12-30`) to BF16 (`1.47 GB`) and compute readout lazily at span/trajectory end (with `record_latent_readout=False` fast path). | **M2** | `survey_1` (`OPT-5`, `OPT-6`, `OPT-7`), `survey_2`, `survey_3` |
| **F6** | **Multimodal Vision Support (`Gemma4ForConditionalGeneration` + `AutoProcessor`)** | Update `load_hybrid_agent` and `HybridAgentModel` (`hybrid_agent/model.py:111-221`) to preserve `vision_tower` / `embed_vision` on `Gemma4ForConditionalGeneration`, load/expose `processor` (`AutoProcessor` / `Gemma4Processor` with fallback to `google/gemma-4-26B-A4B-it` or constructed image processor when `processor_config.json` is absent in `milkshake721/Gemma-4-26B-A4B-HybridAgent`), and pass multimodal `images` / `pixel_values` into `HybridGenerationEngine._init_generation` and `HybridEngine.run` / `run_batch`. | **M2** | `survey_1` (`OPT-8`), `survey_2`, `survey_3` |
| **F7** | **High-Concurrency CLI & Batch Runner Upgrades** | Upgrade `hybrid_inference/run_batch.py`, `sglang_tensorrt_runner.py`, and `scripts/run_batch_inference.sh` to support `--backend {tensorrt,eager,sglang}` (default `tensorrt`), `--question_batch_size` / `--concurrency` (intra-GPU multi-question batching), `--no_tools`, and high-concurrency throughput metrics (`tok/s`, `questions/s`, peak VRAM, latency percentiles). | **M3** | `survey_1` (`OPT-4`), `survey_2`, `survey_3` |
| **F8** | **5-Benchmark HuggingFace Evaluation Suite on G4 GPU (`MMLU Pro`, `AIME 2026 No Tools`, `GPQA Diamond`, `LiveCodeBench v6`, `MMMU Pro`)** | Build `hybrid_agent/benchmarks/` (`datasets.py`, `prompts.py`, `graders.py`, `runner.py`), `eval_hf_benchmarks.py`, and upgraded `eval_gemma4_26b_hybrid.py` supporting `milkshake721/Gemma-4-26B-A4B-HybridAgent` (and `-4bit`) on G4 GPU (`NVIDIA RTX PRO 6000 Blackwell` 96GB) with high-concurrency TensorRT execution across all 5 benchmarks (`MMLU Pro`: 82.6%, `AIME 2026 No Tools`: 88.3%, `GPQA Diamond`: 82.3%, `LiveCodeBench v6`: 77.1%, `MMMU Pro`: 73.8%), including offline/local dataset fallback, strict no-tools enforcement on AIME 2026, subprocess sandbox grading for LiveCodeBench v6, multimodal image evaluation for MMMU Pro, and comprehensive report generation. | **M3** | `survey_1` (`OPT-9`), `survey_2`, `survey_3` |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| **M1** | **High-Concurrency TensorRT & Zero-Sync MoE Backend Acceleration** | Implement **F1, F2, F3** in `hybrid_agent/backends.py` and `hybrid_agent/quantization.py` (plus unit tests in `tests/unit/test_m1_backends.py`). | none | `DONE` (`sub_orch_m1_1` PASS: 15/15 unit tests, 2 Reviewers APPROVE, 2 Challengers PASS, Auditor CLEAN) |
| **M2** | **High-Concurrency Multi-Question Engine, Vectorized Stepping & Multimodal Support** | Implement **F4, F5, F6** in `hybrid_agent/engine.py`, `hybrid_agent/sampling.py`, `hybrid_agent/readout.py`, `hybrid_agent/model.py`, and `hybrid_agent/__init__.py` (plus unit tests in `tests/unit/test_m2_engine.py`). | none | `DONE` (`sub_orch_m2_1` PASS: 12/12 unit tests, 2 Reviewers APPROVE, 2 Challengers PASS, Auditor CLEAN) |
| **M3** | **High-Concurrency Runners & 5-Benchmark HuggingFace Evaluation Suite on G4 GPU** | Implement **F7, F8** in `hybrid_agent/benchmarks/*`, `eval_hf_benchmarks.py`, `eval_gemma4_26b_hybrid.py`, `hybrid_inference/run_batch.py`, `sglang_tensorrt_runner.py`, `scripts/run_batch_inference.sh`, and update `README.md`, `BENCHMARK_AND_OPTIMIZATION_NOTES.md`, and root `PROJECT.md` (plus unit tests in `tests/unit/test_m3_benchmarks.py`). | M1, M2 | `IN_PROGRESS` |
| **E2E** | **Requirement-Driven 4-Tier E2E Test Suite Track** | Create `TEST_INFRA.md`, `tests/e2e/*` (Tiers 1–4 covering F1–F8, ≥98 test cases), `tests/run_e2e_tests.py`, and publish `TEST_READY.md` at project root. | none | `IN_PROGRESS` |
| **M4** | **Final Milestone: Phase 1 E2E Test Pass (Tiers 1–4) & Phase 2 Adversarial Coverage Hardening (Tier 5)** | Phase 1: Pass 100% of E2E test suite (Tiers 1→2→3→4). Phase 2: White-box adversarial coverage hardening (Tier 5) via Challenger (`test-coverage-audit`) → Worker → Reviewer loop until 0 gaps remain. | M1, M2, M3, E2E | `PLANNED` |

---

## Interface Contracts

### 1. `hybrid_agent/backends.py` & `hybrid_agent/quantization.py` (M1) ↔ `hybrid_agent/model.py` & `hybrid_agent/engine.py` (M2)

```python
# hybrid_agent/backends.py
class _TensorRTHybridLinkWrapper(nn.Module):
    def __init__(
        self,
        base_link: nn.Module,
        trt_engine_2d: Any,
        hidden_size: int,
        max_batch_size: int = 256,
    ) -> None: ...
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Accepts h of shape [B, H] or [B, S, H] for ANY batch size B >= 1.
        Routes 1 <= N <= max_batch_size directly through trt_engine_2d (and chunks if N > max_batch_size)
        without falling back to eager when B > 1.
        """

def compile_hybridlink_tensorrt(
    hybridlink: nn.Module,
    hidden_size: int = 2816,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    min_batch_size: int = 1,
    opt_batch_size: int = 32,
    max_batch_size: int = 256,
) -> nn.Module: ...

def patch_bf16_experts_fastpath(backbone: nn.Module) -> int:
    """Patches all Gemma4TextExperts layers with zero-sync batched/grouped MoE forward
    that supports arbitrary token counts n_tokens >= 1 (including B > 8) with zero .nonzero() CPU-GPU syncs.
    """

def patch_compiled_ffn_bf16(backbone: nn.Module) -> int:
    """Compiles post-attention RMSNorm + Shared MLP + Router + BF16 MoE + 6 RMSNorms via torch.compile(dynamic=True)."""

def enable_tensorrt_acceleration(
    model: Any,
    *,
    compile_backbone_layers: bool = True,
    use_fp8_or_bf16: str = "bf16",
    max_batch_size: int = 256,
    opt_batch_size: int = 32,
) -> Any: ...
```

### 2. `hybrid_agent/model.py` & `hybrid_agent/engine.py` (M2) ↔ `hybrid_agent/benchmarks/*` & CLI Runners (M3)

```python
# hybrid_agent/model.py
def load_hybrid_agent(
    path_or_repo: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    trust_remote_code: bool = False,
    require_hybridlink: bool = True,
    revision: str | None = None,
    backend: str = "eager",
    max_batch_size: int = 256,
    load_processor: bool = False,
) -> tuple[HybridAgentModel, PreTrainedTokenizerBase]:
    """Loads HybridAgentModel and tokenizer; also attaches model.processor (Gemma4Processor / AutoProcessor
    with automatic fallback when processor_config.json is absent) for multimodal inputs.
    """

# hybrid_agent/engine.py
class HybridGenerationEngine:
    def __init__(
        self,
        model: HybridAgentModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str | Sequence[str],
        *,
        images: Any | Sequence[Any] | None = None,
        processor: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        batch_size: int = 1,
        max_response_tokens: int = 8192,
        max_latent_steps: int = 32,
        min_latent_steps: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_rounds: int = 8,
        max_tool_calls: int = 6,
        max_python_calls: int = 6,
        max_search_calls: int = 6,
        tool_executor: ToolExecutor | None = None,
        record_hidden: bool = False,
        record_token_logprobs: bool = False,
        record_latent_readout: bool = False,
    ) -> None:
        """If `question` is a Sequence[str] of length M, batches M distinct questions
        (with left-padded prefill and optional per-question `images`), setting B = M (or M * batch_size).
        If `question` is a single `str`, preserves backward-compatible B = batch_size rollouts.
        """

class HybridEngine:
    def __init__(
        self,
        model: HybridAgentModel,
        tokenizer: PreTrainedTokenizerBase,
        tool_executor: ToolExecutor | None = None,
        *,
        processor: Any | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 4096,
        max_latent_steps: int = 32,
        min_latent_steps: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_rounds: int = 6,
        max_tool_calls: int = 6,
        max_python_calls: int = 6,
        max_search_calls: int = 6,
        record_latent_readout: bool = True,
        default_batch_size: int = 32,
    ) -> None: ...

    def run(self, question: str, *, images: Any | None = None, **overrides: Any) -> HybridTrajectory: ...

    def run_batch(
        self,
        questions: Sequence[str],
        *,
        images: Sequence[Any] | None = None,
        batch_size: int | None = None,
        max_concurrency: int | None = None,
        record_latent_readout: bool | None = None,
        **overrides: Any,
    ) -> list[HybridTrajectory]:
        """Executes a list of distinct questions (and optional multimodal images) in high-concurrency
        batches of size `batch_size`, preserving input order in the returned `list[HybridTrajectory]`.
        """
```

### 3. `hybrid_agent/benchmarks/` & `eval_hf_benchmarks.py` (M3) ↔ E2E Test Suite (E2E)

```python
# hybrid_agent/benchmarks/__init__.py
BENCHMARK_SPECS: dict[str, BenchmarkSpec]  # Keys: "mmlu_pro", "aime_2026", "gpqa_diamond", "livecodebench_v6", "mmmu_pro"
REFERENCE_SCORES: dict[str, float] = {
    "mmlu_pro": 82.6,
    "aime_2026": 88.3,
    "gpqa_diamond": 82.3,
    "livecodebench_v6": 77.1,
    "mmmu_pro": 73.8,
}

def recommend_g4_concurrency(
    model_id_or_path: str,
    vram_gb: float = 96.0,
    quantized_4bit: bool | None = None,
) -> dict[str, int]:
    """Returns recommended batch_size / max_concurrency for G4 GPU (96GB RTX PRO 6000 Blackwell)
    or smaller GPUs based on model precision and benchmark sequence length.
    """

def evaluate_benchmark(
    benchmark_name: str,
    engine: HybridEngine,
    *,
    dataset_items: Sequence[dict[str, Any]] | None = None,
    split: str | None = None,
    limit: int | None = None,
    batch_size: int = 32,
    output_path: str | Path | None = None,
) -> BenchmarkResult: ...
```

CLI invocation contract for `eval_hf_benchmarks.py`:
```bash
python3 eval_hf_benchmarks.py \
  --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
  --backend tensorrt \
  --benchmarks all \
  --batch-size 32 \
  --output-dir outputs/benchmarks
```
Supports `--benchmarks {all,mmlu_pro,aime_2026,gpqa_diamond,livecodebench_v6,mmmu_pro}`, `--limit N`, `--dataset-dir /path/to/local/datasets`, `--dry-run / --synthetic-smoke` (for CPU/offline CI verification without a 96GB GPU), and outputs a JSON + Markdown report comparing achieved scores against the 5 reference scores (`82.6%`, `88.3%`, `82.3%`, `77.1%`, `73.8%`) along with TensorRT concurrency throughput metrics.

---

## Code Layout

```text
/home/antigravity/milkshake0721/HybridAgent/
├── PROJECT.md                              # Global project architecture & milestone index (mirrored from .agents/teamwork)
├── TEST_INFRA.md                           # E2E test infrastructure specification
├── TEST_READY.md                           # E2E test suite completion signal & coverage matrix
├── eval_hf_benchmarks.py                   # Unified high-concurrency TensorRT CLI for all 5 HF benchmarks on G4 GPU
├── eval_gemma4_26b_hybrid.py               # Upgraded Gemma-4 26B-A4B hybrid benchmark script (supports batched TensorRT & 5 HF benchmarks)
├── sglang_tensorrt_runner.py               # Single & multi-question high-concurrency TensorRT / SGLang CLI runner
├── test_gemma4_26b_hybrid.py               # Smoke test script
├── hybrid_agent/
│   ├── __init__.py                         # Public package exports
│   ├── backends.py                         # [M1] Dynamic-batch TensorRT HybridLink, zero-sync BF16 MoE, compiled BF16/4bit FFN, SGLangHybridEngine
│   ├── quantization.py                     # [M1] Pack-quantized 4-bit loader & expert-grouped Triton W4A16 MoE kernel
│   ├── engine.py                           # [M2] Multi-question batched HybridGenerationEngine, HybridEngine.run_batch, multi-token FEED
│   ├── sampling.py                         # [M2] Vectorized greedy/multinomial sampling & KV cache batch utilities
│   ├── readout.py                          # [M2] BF16 normalized embedding bank & lazy latent readout
│   ├── model.py                            # [M2] HybridAgentModel, load_hybrid_agent, multimodal AutoProcessor / Gemma4Processor loader
│   ├── answer.py                           # Answer extraction utilities
│   ├── constants.py                        # Special token constants
│   ├── hybridlink.py                       # HybridLink adapter module
│   ├── prompts.py                          # System prompts (tool-enabled & no-tools variants)
│   ├── tools/                              # Python & Tavily search tool executors
│   └── benchmarks/                         # [M3] 5-Benchmark HuggingFace evaluation package
│       ├── __init__.py
│       ├── datasets.py                     # Loaders & normalizers for MMLU Pro, AIME 2026, GPQA Diamond, LiveCodeBench v6, MMMU Pro
│       ├── prompts.py                      # Benchmark-specific prompt & tool-policy builders
│       ├── graders.py                      # Exact letter, math_verify 000-999, and LiveCodeBench v6 sandbox test-execution graders
│       └── runner.py                       # High-concurrency batched TensorRT benchmark runner, G4 GPU auto-tuner & telemetry reporter
├── hybrid_inference/
│   └── run_batch.py                        # [M3] Multi-question batched JSONL runner with --backend tensorrt & --question_batch_size
└── tests/
    ├── run_e2e_tests.py                    # [E2E] Unified E2E test runner (exit code 0 on 100% pass)
    ├── unit/                               # [M1, M2, M3] Module unit tests
    └── e2e/                                # [E2E, M4] Tiers 1-4 opaque-box E2E tests + Tier 5 adversarial tests
```
