# Gemma-4-26B-A4B-HybridAgent — TensorRT & Triton W4A16 隱空間推論加速架構說明

本文件說明 [`milkshake721/Gemma-4-26B-A4B-HybridAgent`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent)（BF16 原生版）與 [`milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit`](https://huggingface.co/milkshake721/Gemma-4-26B-A4B-HybridAgent-4bit)（4-bit `compressed-tensors` W4A16 量化版）在 **HybridAgent 隱空間切換（`<latent>...</latent>`）與多回合工具呼叫（`<python>...</python>`、`<search>...</search>`）** 架構下的核心推論機制、算子融合設計與基準測試數據。

---

## 1. 混合推論狀態機（Hybrid Generation State Machine）

`Gemma-4-26B-A4B-HybridAgent` 的推論引擎在單一持續性 KV Cache（Working Memory）上動態切換三種執行模式：

1. **文字模式 (`DECODE`)**：標準離散 Token 抽樣，直到生成進入隱空間的特殊錨點 `<latent>`（`token_id = 262144`）。
2. **隱空間連續思考模式 (`LATENT`)**：
   - 當模型輸出 `<latent>` 時，推論引擎擷取最後一層隱藏態 $h_t \in \mathbb{R}^{2816}$，送入輕量級殘差橋接網路 **`HybridLink`（`adapter.pt` / `hybridlink.pt`，2 層 GELU Residual MLP + LayerNorm，15.87M 參數）**：
     $$e_{t+1} = \mathcal{R}(h_t) \cdot \sqrt{d_{\text{model}}} \quad (\sqrt{2816} \approx 53.0660)$$
   - 直接以 `inputs_embeds = e_{t+1}` 送入下一步 Decoder，在連續隱空間中遞迴推演，直到最近鄰餘弦相似度讀出（Readout）對齊離開錨點 `</latent>`（`token_id = 262145`）後切回文字模式。
3. **工具互動模式 (`FEED`)**：攔截 `<python>...</python>` 或 `<search>...</search>`，於沙盒執行後將 `<result>...</result>` 寫回同一份 KV Cache 延續多回合推論。

---

## 2. 核心加速架構與算子最佳化

針對 `google/gemma-4-26B-A4B`（30 層 Decoder，每層含 1 個共享 Dense MLP 與 128 個 Routed Experts，每 Token 動態激活 Top-8 專家），`backend="tensorrt"` 實作了三項底層加速：

### 2.1 零同步批次專家路由（Zero-Sync Batched MoE Fast Path，BF16 版）
- **設計背景**：`transformers` 預設的 `Gemma4TextExperts.forward` 透過 `.nonzero()` 動態篩選被命中的專家並以 Python 迴圈逐一計算，在單步解碼（`T = 1`）時每層皆會觸發 CPU-GPU 同步阻塞（30 層共 30 次同步與 240 次迴圈）。
- **最佳化實作（`patch_bf16_experts_fastpath`）**：針對 `T <= 8` 的生成步，直接以 `top_k_index` 進行 3D 張量索引與批次矩陣乘法（`torch.mv` / `torch.bmm`），達成 **0 次 CPU-GPU 同步、0 次 Python 迴圈**，將 30 層 BF16 MoE 單步耗時壓低至 **`4.60 ms / token`**。

### 2.2 暫存器級融合 Triton W4A16 MoE Kernel（Register-Fused Triton W4A16，4-bit 版）
- **設計背景**：傳統 W4A16 若先將 INT4 權重反量化為 BF16 寫回 HBM 顯存再呼叫 cuBLAS，每層會產生 `229.5 MB` 的顯存讀寫流量，受限於 HBM 頻寬瓶頸。
- **最佳化實作（`_fused_w4a16_moe_kernel`）**：以自訂 Triton Kernel 在 **GPU 暫存器（Registers / L1 SRAM）** 內一次完成：
  1. 載入 `int32` 打包權重與 `bf16` Group-32 Scales；
  2. 位移解包 `((packed >> shifts) & 0xF) - 8` 並對齊 `bfloat16` 縮放（`w_bf16 = q_bf16 * scales_bf16`）；
  3. 於暫存器內直接與輸入激活向量 $x$ 完成內積累加，**不將解壓後的 BF16 權重寫回 HBM**。
- **加速效果**：30 層 4-bit MoE 單步耗時僅 **`2.02 ms / token`**（較 BF16 的 `4.60 ms` 快 **2.28 倍**）。

### 2.3 C++ 融合 `RMSNorm` + 整圖編譯 FFN + TensorRT `HybridLink`
- **C++ 融合正規化（`_fast_rmsnorm_forward`）**：將每層 11 個 `Gemma4RMSNorm`（30 層共 330 個）由原本的 8 個逐元素 Python 算子替換為單一 C++ 融合算子 `torch.nn.functional.rms_norm`，消除每步 2,640 次微小 CUDA Kernel 派發開銷。
- **後注意力區塊整圖編譯（`patch_compiled_ffn_4bit`）**：將 `fused_triton_4bit_moe_forward` 註冊為 `@torch.library.custom_op("hybrid_agent::fused_4bit_moe")`，並將每層 Attention 之後的 `RMSNorm + Shared Dense MLP + Router + Triton 4-bit MoE` 透過 `torch.compile(fullgraph=True, dynamic=True)` 編譯為單一融合計算圖。
- **TensorRT 橋接器（`compile_hybridlink_tensorrt`）**：透過 `torch_tensorrt.compile` 將 `HybridLink` 編譯為原生 TensorRT `ICudaEngine`。

---

## 3. 基準測試結果（Benchmark Results）

- **測試硬體**：NVIDIA RTX PRO 6000 Blackwell Server Edition (`CUDA 13.0`, `PyTorch 2.11`)
- **評測腳本**：[`eval_gemma4_26b_hybrid.py`](eval_gemma4_26b_hybrid.py)（10 題數學與邏輯推理基準測試，`max_new_tokens=1024`, `max_latent_steps=64`, `temperature=0.0`）

### 3.1 總體效能對比

| 模型與推論後端 | 常駐 VRAM | 峰值 VRAM | 模型載入時間 | Prefill (180 tok) | Single-Token Decode | 10 題總耗時 | 平均每題耗時 | 10 題準確率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma-4-26B-A4B-HybridAgent` (BF16, `backend="tensorrt"`)** | `54.73 GB` | `60.64 GB` | **`8.80s`** | `313.03 ms` | `26.01 ms` (`38.5 tok/s`) | `106.80s` | `10.68s` | **`10/10 (100%)`** |
| 🏆 **`Gemma-4-26B-A4B-HybridAgent-4bit` (W4A16, `backend="tensorrt"`)** | **`19.00 GB`** | **`24.93 GB`** | `29.69s` | **`187.51 ms`** | **`16.62 ms` (`60.2 tok/s`)** | **`83.65s`** | **`8.36s`** | **`10/10 (100%)`** |

### 3.2 逐題推論表現（Q01 ～ Q10）

| 題號 | 題目摘要 | 標準答案 | `BF16` (`backend="tensorrt"`) | 🏆 **`4-bit W4A16` (`backend="tensorrt"`)** |
| :---: | :--- | :---: | :---: | :---: |
| **Q01** | `If 3x + 7 = 22, what is 5x - 2?` | `23` | ✅ `9.71s` (`0` latent, `1` py) | ✅ **`6.43s`** (`0` latent, `1` py) |
| **Q02** | `Sum of integers from 1 to 20` | `210` | ✅ `6.30s` (`74` latent, `1` py) | ✅ **`4.52s`** (`17` latent, `1` py) |
| **Q03** | `Rectangle perimeter 30, area 54, longer side?` | `9` | ✅ `12.64s` (`50` latent, `1` py) | ✅ **`19.68s`** (`128` latent, `1` py) |
| **Q04** | `Distinct rearrangements of 'LEVEL'` | `30` | ✅ `14.23s` (`64` latent, `1` py) | ✅ **`7.77s`** (`64` latent, `1` py) |
| **Q05** | `f(x) = x^2 - 3x + 4, f(f(2)) = ?` | `2` | ✅ `7.39s` (`37` latent, `1` py) | ✅ **`6.04s`** (`0` latent, `1` py) |
| **Q06** | `Train 120km @ 60km/h + 120km @ 40km/h avg speed` | `48` | ✅ `13.17s` (`135` latent, `1` py) | ✅ **`9.92s`** (`0` latent, `1` py) |
| **Q07** | `Remainder when 2^10 is divided by 7` | `2` | ✅ `7.27s` (`82` latent, `1` py) | ✅ **`4.00s`** (`0` latent, `1` py) |
| **Q08** | `Triangle angles 45° and 65°, third angle?` | `70` | ✅ `5.44s` (`37` latent, `1` py) | ✅ **`5.60s`** (`0` latent, `1` py) |
| **Q09** | `Positive solution to x^2 + 4x - 21 = 0` | `3` | ✅ `10.01s` (`0` latent, `1` py) | ✅ **`9.73s`** (`0` latent, `1` py) |
| **Q10** | `Three coins probability of exactly two heads` | `3/8` | ✅ `20.64s` (`122` latent, `1` py) | ✅ **`9.94s`** (`15` latent, `1` py) |
| **總計** | **10 題基準測試** | **10/10** | **`106.80s` (100%)** | 🏆 **`83.65s` (100%)** |

---

## 4. 精度與硬體規格選型對照

| 部署方案 | 權重精度 | 激活值精度 | 常駐 VRAM (26B MoE) | 硬體執行方式 | 建議硬體與適用場景 |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **`BF16` 原生版** (`Gemma-4-26B-A4B-HybridAgent`) | `BF16` (16-bit) | `BF16` (16-bit) | `54.73 GB` | 原生 BF16 Tensor Core + Zero-Sync MoE | **80GB+ 顯卡（A100 / H100 / RTX PRO 6000）**；載入速度最快（`6~8s`），無損全精度基準。 |
| **`W4A16` 4-bit 版** (`Gemma-4-26B-A4B-HybridAgent-4bit`) | `INT4` (Group-32) | `BF16` (16-bit) | **`19.00 GB`** | **Register-Fused Triton Kernel**（暫存器內 INT4 解包 $\rightarrow$ BF16 內積） | **24GB+ 顯卡（RTX 4090 / 3090 / L4）首選**；顯存節省 65%，Single-Token 達 **`60.2 tok/s`**、Prefill `187.5 ms`，兼具高吞吐與無損推理準確率。 |
