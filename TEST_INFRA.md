# HybridAgent E2E Test Infrastructure (`TEST_INFRA.md`)

## 1. Overview & Architecture

The HybridAgent End-to-End (E2E) Test Suite (`tests/e2e/`) provides a **104-test, 4-Tier opaque-box verification harness** covering all 8 features (`F1`–`F8`) across Milestones M1, M2, and M3:

| Tier | File | Test Count | Focus |
|---|---|---:|---|
| **Tier 1** | [`tests/e2e/test_tier1_features.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/e2e/test_tier1_features.py) | **40** (`5 × F1..F8`) | Happy-path functional verification of each feature's public contract |
| **Tier 2** | [`tests/e2e/test_tier2_boundaries.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/e2e/test_tier2_boundaries.py) | **40** (`5 × F1..F8`) | Boundary values, edge cases, fallback chains, malformed inputs, and security/cap guardrails |
| **Tier 3** | [`tests/e2e/test_tier3_combinations.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/e2e/test_tier3_combinations.py) | **16** (`test_t3_c01..c16`) | Cross-feature and cross-milestone (`M1 × M2 × M3`) interaction matrices |
| **Tier 4** | [`tests/e2e/test_tier4_scenarios.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/e2e/test_tier4_scenarios.py) | **8** (`test_t4_s01..s08`) | Full user journeys from checkpoint persistence and G4 GPU auto-tuning through multi-shard crash recovery and all 4 CLI entrypoints |
| **Total** | [`tests/run_e2e_tests.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/run_e2e_tests.py) | **104** | Unified CLI runner with per-tier filtering and JSON reporting |

---

## 2. Dual-Environment Execution Strategy

The test infrastructure in [`tests/e2e/helpers.py`](file:///home/antigravity/milkshake0721/HybridAgent/tests/e2e/helpers.py) is engineered to run identically in two environments:

1. **CPU Sandbox / CI Containers (`torch 2.15.0+cpu`, `transformers 5.22.2`, `safetensors 0.8.0`)**:
   - `install_runtime_compat()` configures `HF_HUB_OFFLINE=1` by default, redirects `device="cuda"` calls transparently to `"cpu"` when `not torch.cuda.is_available()`, and provides a fast CPU `torch.compile` wrapper and `sitecustomize.py` bootstrap for child CLI subprocesses.
   - `ScriptedHybridBackbone` (`TinyGemma4Backbone`) instantiates genuine PyTorch `Gemma4TextExperts` (BF16) and `Gemma4TextExperts4Bit` (`Linear4Bit` W4A16) MoE decoder layers, `vision_tower`, and `embed_vision` modules while driving deterministic per-row `<think>`, `<latent>`, `<python>`, `<search>`, `<result>`, and `<answer>\boxed{...}</answer>` token trajectories.
2. **Google Cloud G4 GPU (`NVIDIA RTX PRO 6000 Blackwell` 96GB GDDR7, CUDA `sm_120`, `torch_tensorrt`)**:
   - Exercises hardware TensorRT engines (`min=1, opt=32, max=256`), fused Triton BF16 & W4A16 MoE kernels, and full `Gemma-4-26B-A4B-HybridAgent` / `Gemma-4-26B-A4B-HybridAgent-4bit` checkpoints across the 5 HuggingFace benchmarks.

---

## 3. Shared Fixtures & Helpers (`tests/e2e/helpers.py`)

- **`install_runtime_compat()` & `ensure_contract_bridges()`**:
  - Automatically delegates 100% to production M1 (`hybrid_agent.backends`, `hybrid_agent.quantization`), M2 (`hybrid_agent.engine`, `hybrid_agent.model`, `hybrid_agent.sampling`, `hybrid_agent.readout`), and M3 (`hybrid_agent.benchmarks`, `eval_hf_benchmarks.py`, `hybrid_inference/run_batch.py`, `sglang_tensorrt_runner.py`, `eval_gemma4_26b_hybrid.py`) modules on disk, while providing contract bridges during parallel milestone development.
- **`SyntheticGemma4Tokenizer` & `build_tiny_tokenizer(vocab_size=544)`**:
  - Lossless byte-level + atomic special-token (`<latent>`, `</latent>`, `<think>`, `</think>`, `<python>`, `</python>`, `<search>`, `</search>`, `<result>`, `</result>`, `<answer>`, `</answer>`, `<|image>`, `<|image|>`, `<image|>`) tokenizer supporting left-padding (`padding_side="left"`), `apply_chat_template`, and split closing tag tokens (`"</py"` + `"thon>\n"`).
- **`SpyTensorRTEngine`**:
  - Wraps a `HybridLink` module to record 2D batch invocations (`call_count`, `call_shapes`, `call_sizes`) inside `_TensorRTHybridLinkWrapper`.
- **`ScriptedHybridBackbone` & `build_tiny_hybrid_model(...)`**:
  - Miniature Gemma 4 MoE `nn.Module` supporting BF16 (`Gemma4TextExperts`) and 4-bit (`Gemma4TextExperts4Bit` + `Linear4Bit`), multimodal vision (`vision_tower` + `embed_vision`), and per-row state tracking across `expand_cache_batch`, `_lockstep_forward`, and `_multi_token_feed_step`.
- **`create_temp_checkpoint_dir(...)` & `create_mini_hybrid_checkpoint(...)`**:
  - Generates local checkpoint directories with `config.json`, `model.safetensors`, `adapter.pt`, `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json`, and optional `processor_config.json`.
- **`build_synthetic_benchmark_items()` & `create_synthetic_benchmark_datasets(tmp_path)`**:
  - Builds offline synthetic items and JSONL/PNG datasets for all 5 target benchmarks:
    1. `mmlu_pro` (`TIGER-Lab/MMLU-Pro`, 10 choices `A`–`J`, reference `82.6%`)
    2. `aime_2026` (`MathArena/aime_2026`, strict 0-tool integer `000`–`999`, reference `88.3%`)
    3. `gpqa_diamond` (`Idavidrein/gpqa` / `fingertap/GPQA-Diamond`, 4 choices `A`–`D`, reference `82.3%`)
    4. `livecodebench_v6` (`livecodebench/code_generation_lite`, `stdin` & `starter_code` functional + `base64(zlib(pickle(json)))` private tests, reference `77.1%`)
    5. `mmmu_pro` (`MMMU/MMMU_Pro`, `standard (10 options)` multi-image & `vision` subsets, reference `73.8%`)

---

## 4. Running the E2E Test Suite

```bash
# Run all 104 E2E tests across Tiers 1, 2, 3, and 4
python3 tests/run_e2e_tests.py --tier all -v

# Run individual tiers
python3 tests/run_e2e_tests.py --tier 1 -v
python3 tests/run_e2e_tests.py --tier 2 -v
python3 tests/run_e2e_tests.py --tier 3 -v
python3 tests/run_e2e_tests.py --tier 4 -v

# Filter by substring and save JSON report
python3 tests/run_e2e_tests.py --tier all -k mmmu_pro --json-report outputs/e2e_report.json
```
