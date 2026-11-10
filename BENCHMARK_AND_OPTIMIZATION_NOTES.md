# Gemma-4-26B-A4B-HybridAgent — High-Concurrency TensorRT & Triton W4A16 隱空間推論加速與 G4 GPU 五大基準評測說明

本文件說明 [`milkshake721/Gemma-4-26B-A4B-HybridAgent`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent)（BF16 原生版）與 [`milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit)（4-bit `compressed-tensors` W4A16 量化版）在 **HybridAgent 隱空間切換（`<latent>...</latent>`）、多回合工具呼叫（`<python>...</python>`、`<search>...</search>`）與多模態視覺輸入（`Gemma4ForConditionalGeneration`）** 架構下的高併發 TensorRT 算子融合設計，以及在 **Google Cloud G4 GPU（`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7 VRAM）** 上執行五大 HuggingFace 權威基準測試（`MMLU Pro`、`AIME 2026 No Tools`、`GPQA Diamond`、`LiveCodeBench v6`、`MMMU Pro`）的完整架構。

---

## 1. 混合推論狀態機與高併發批次引擎（Multi-Question Hybrid Generation Engine）

`Gemma-4-26B-A4B-HybridAgent` 的推論引擎（`HybridGenerationEngine` 與 `HybridEngine.run_batch`）支援將 $M$ 道不同問題（含不同長度的 Prompt 與多模態圖片）合併於單一持續性 KV Cache（Working Memory）中進行鎖步（Lockstep）高併發推論，並動態切換三種執行模式：

1. **文字模式 (`DECODE`)**：
   - 支援左填充（Left-Padded）多問題批次 Prefill 與預分配 `_attn_mask_buf`，消除每步 `torch.ones` 顯存配置。
   - 向量化 Greedy (`argmax`) 與 Stochastic (`multinomial`) 抽樣，單次 `.tolist()` 批次轉移取代逐行 `.item()` CPU-GPU 同步，並僅在輸出 `>` 字元時才觸發 `_row_tail` 標籤檢測。
2. **隱空間連續思考模式 (`LATENT`)**：
   - 當任一軌跡輸出 `<latent>`（`token_id = 262144`）時，推論引擎擷取最後一層隱藏態 $h_t \in \mathbb{R}^{B_{\text{lat}} \times 2816}$，送入動態批次 **TensorRT `HybridLink`（`adapter.pt` / `hybridlink.pt`，2 層 GELU Residual MLP + LayerNorm，15.87M 參數）**：
     $$e_{t+1} = \mathcal{R}(h_t) \cdot \sqrt{d_{\text{model}}} \quad (\sqrt{2816} \approx 53.0660)$$
   - 採用 **BF16 正規化嵌入庫（`1.47 GB`，省去 50% FP32 顯存）** 進行向量化餘弦相似度計算，並在單次 GPU 矩陣乘法中同時完成 `</latent>` 離開錨點檢測與 Top-1 偽文字讀出（Lazy Readout）。
3. **多 Token 批次工具回填模式 (`FEED`)**：
   - 攔截 `<python>...</python>` 或 `<search>...</search>`，於沙盒並行執行後，透過多 Token 分塊（Chunked Prefill，`_MAX_FEED_CHUNK = 512`）一次將 `<result>...</result>` 寫回對應行的 KV Cache，大幅降低長工具觀測值的回填延遲。

---

## 2. 高併發底層加速架構與算子最佳化

針對 `google/gemma-4-26B-A4B`（30 層 Decoder，每層含 1 個共享 Dense MLP 與 128 個 Routed Experts，每 Token 動態激活 Top-8 專家），`backend="tensorrt"` 實作了四項關鍵高併發加速：

### 2.1 動態批次 TensorRT `HybridLink` 引擎（Dynamic-Batch `_TensorRTHybridLinkWrapper`）
- **最佳化實作（`compile_hybridlink_tensorrt`）**：配置 TensorRT 動態形狀優化範圍 `min_shape=(1, 2816)`、`opt_shape=(32, 2816)`、`max_shape=(256, 2816)`，支援任意批次大小 $B \ge 1$ 直接由原生 TensorRT `ICudaEngine` 執行（當 $N > \text{max\_batch\_size}$ 時自動分塊執行，永不退回慢速 Eager 模式）。

### 2.2 高併發零同步 BF16 專家路由與整圖編譯（Zero-Sync Micro-Batched BF16 MoE & `patch_compiled_ffn_bf16`）
- **設計背景**：`transformers` 預設的 `Gemma4TextExperts.forward` 透過 `.nonzero()` 動態篩選專家並以 Python 迴圈逐一計算，每層觸發 CPU-GPU 同步阻塞；而單純將 `T * K` 專家權重複製展開在 $T > 8$ 高併發下會造成顯存暴增。
- **最佳化實作（`patch_bf16_experts_fastpath` + `patch_compiled_ffn_bf16`）**：
  - 徹底移除 `.nonzero()` CPU-GPU 同步與 `n_tokens <= 8` 限制，對任意 $T \ge 1$ 以零同步微批次（`_BF16_MOE_MICRO_BATCH = 8`）限制峰值工作顯存。
  - 新增 `patch_compiled_ffn_bf16`，將 BF16 模型每層 Attention 之後的 `RMSNorm + Shared Dense MLP + Router + BF16 MoE + 6 RMSNorms` 透過 `torch.compile(dynamic=True)` 編譯為單一融合計算圖。

### 2.3 專家排序 L2 快取局部性最佳化 Triton W4A16 MoE Kernel（4-bit 版）
- **最佳化實作（`_fused_w4a16_moe_kernel` + `sort_moe_token_expert_pairs`）**：
  - 在 GPU 暫存器（Registers / L1 SRAM）內一次完成 INT4 位移解包 `((packed >> shifts) & 0xF) - 8`、Group-32 BF16 縮放與內積累加。
  - 當高併發批次 $T > 8$ 時，自動啟用專家索引排序（`torch.argsort(top_k_index.reshape(-1), stable=True)`），讓命中相同專家的不同問題 Token 共享 L2 Cache 中的 INT4 專家權重，大幅提升高併發吞吐量。

### 2.4 多模態視覺預處理與編碼（`Gemma4ForConditionalGeneration` + `HybridMultimodalProcessor`）
- `load_hybrid_agent(..., load_processor=True)` 完整保留 `Gemma4ForConditionalGeneration` 的 `vision_tower` 與 `embed_vision` 模組，並在模型目錄缺少 `processor_config.json` 時自動建構 `HybridMultimodalProcessor`，支援單圖與多圖（如 `MMMU Pro` 的 `image_1..image_7`）高併發批次視覺編碼。

---

## 3. Google Cloud G4 GPU 五大 HuggingFace 基準評測套件（5-Benchmark Evaluation Suite）

針對 **Google Cloud G4 GPU (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7 VRAM)**，本專案於 [`hybrid_agent/benchmarks/`](hybrid_agent/benchmarks/) 與 [`eval_hf_benchmarks.py`](eval_hf_benchmarks.py) 實作了五大權威基準評測與自動併發調優器（`recommend_g4_concurrency`）：

| 基準測試 (Benchmark) | HuggingFace 資料集 | 題型與評分器 (Grader) | 工具策略 | 目標參考分數 (`Gemma-4-26B-A4B`) | G4 GPU (96GB) 建議併發 Batch Size (BF16 / 4-bit) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **1. MMLU Pro** (`mmlu_pro`) | `TIGER-Lab/MMLU-Pro` (`test`) | 10 選 1（`A`–`J`，自動過濾 `N/A` 選項） | 允許工具 | **`82.6%`** | `B = 64` / `B = 128` |
| **2. AIME 2026 (No Tools)** (`aime_2026`) | `MathArena/aime_2026` | 整數 `000`–`999`（支援前導零與 LaTeX 分數正規化） | **嚴格禁用工具 (`max_tool_calls=0`)** | **`88.3%`** | `B = 32` / `B = 64` |
| **3. GPQA Diamond** (`gpqa_diamond`) | `Idavidrein/gpqa` / `fingertap/GPQA-Diamond` | 4 選 1（`A`–`D`，SHA-256 固定種子確定性洗牌与 Gated Repo 自動 Fallback） | 允許工具 | **`82.3%`** | `B = 32` / `B = 64` |
| **4. LiveCodeBench v6** (`livecodebench_v6`) | `livecodebench/code_generation_lite` (`release_v6`) | `pass@1` 子行程沙盒測試（支援 `base64+zlib+pickle` 私測解壓、`stdin` 與 `Solution` 類別測試） | 允許工具 | **`77.1%`** | `B = 32` / `B = 64` |
| **5. MMMU Pro (多模態)** (`mmmu_pro`) | `MMMU/MMMU_Pro` (`standard (10 options)` & `vision`) | 多模態圖片（`image_1..image_7` 與 `image`）+ 10 選 1（`A`–`J`） | 允許工具 | **`73.8%`** | `B = 32` / `B = 64` |

### 執行指令範例

```bash
# 於 G4 GPU (96GB VRAM) 以 TensorRT 高併發模式執行全部 5 項 HuggingFace 評測
python3 eval_hf_benchmarks.py \
    --model milkshake721/Gemma-4-26B-A4B-HybridAgent \
    --backend tensorrt \
    --benchmarks all \
    --batch-size 32 \
    --output-dir outputs/benchmarks
```

執行完成後將於 `outputs/benchmarks/` 產出各基準明細 `<benchmark>.json`、總表 `summary.json` / `benchmark_results.json` 以及與目標分數（`82.6%`、`88.3%`、`82.3%`、`77.1%`、`73.8%`）對照的 Markdown 報告 `report.md` / `SUMMARY.md`。

---

## 4. 10 題數學與邏輯基準測試結果（RTX PRO 6000 Blackwell）

- **測試硬體**：NVIDIA RTX PRO 6000 Blackwell Server Edition (`CUDA 13.0`, `PyTorch 2.11`, 96 GB GDDR7)
- **評測腳本**：[`eval_gemma4_26b_hybrid.py`](eval_gemma4_26b_hybrid.py)（支援高併發批次 `engine.run_batch` TensorRT 執行，`max_new_tokens=1024`, `max_latent_steps=64`, `temperature=0.0`）

### 4.1 總體效能對比

| 模型與推論後端 | 常駐 VRAM | 峰值 VRAM | 模型載入時間 | Prefill (180 tok) | Single-Token Decode | 10 題總耗時 | 平均每題耗時 | 10 題準確率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma-4-26B-A4B-HybridAgent` (BF16, `backend="tensorrt"`)** | `54.73 GB` | `60.64 GB` | **`8.80s`** | `313.03 ms` | `26.01 ms` (`38.5 tok/s`) | `106.80s` | `10.68s` | **`10/10 (100%)`** |
| 🏆 **`Gemma-4-26B-A4B-HybridAgent-4bit` (W4A16, `backend="tensorrt"`)** | **`19.00 GB`** | **`24.93 GB`** | `29.69s` | **`187.51 ms`** | **`16.62 ms` (`60.2 tok/s`)** | **`83.65s`** | **`8.36s`** | **`10/10 (100%)`** |

---

## 5. 精度與硬體規格選型對照

| 部署方案 | 權重精度 | 激活值精度 | 常駐 VRAM (26B MoE) | 硬體執行方式 | 建議硬體與適用場景 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`BF16` 原生版** (`Gemma-4-26B-A4B-HybridAgent`) | `BF16` (16-bit) | `BF16` (16-bit) | `54.73 GB` | 原生 BF16 Tensor Core + Zero-Sync Micro-Batched MoE + Compiled FFN | **Google Cloud G4 GPU (`RTX PRO 6000 Blackwell` 96GB) / H100 / A100**；支援 `B = 32–64` 高併發無損全精度評測。 |
| **`W4A16` 4-bit 版** (`Gemma-4-26B-A4B-HybridAgent-4bit`) | `INT4` (Group-32) | `BF16` (16-bit) | **`19.00 GB`** | **Expert-Sorted Register-Fused Triton Kernel**（暫存器內 INT4 解包 $\rightarrow$ BF16 內積） | **24GB+ 顯卡（RTX 4090 / L4）及 96GB G4 GPU 超高併發（`B = 64–128`）首選**；顯存節省 65%，兼具高吞吐與無損推理準確率。 |
