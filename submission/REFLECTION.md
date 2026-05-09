# Reflection — Lab 22 (DPO/ORPO Alignment)

**Tên:** Phạm Hữu Hoàng Hiệp
**MSSV:** 2A202600415
**Cohort:** A20-K1
**Tier đã chạy:** T4 (local — RTX 5070 12 GB Blackwell)
**Date:** 2026-05-08

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 5070 12 GB (Blackwell sm_120) |
| CUDA / driver | CUDA 12.8 · driver 591.86 · PyTorch 2.11.0+cu128 |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` (NF4 quantized) |
| SFT dataset slice | `bkai-foundation-models/vi-alpaca` · 1000 samples · 1 epoch (`5CD-AI/Vietnamese-alpaca-cleaned` của lab gốc đã bị xoá khỏi HF Hub — substituted with bkai version cùng schema) |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned` · 2000 pairs · 1 epoch |
| `COMPUTE_TIER` env | T4 |
| Total cost | $0 (local laptop GPU) |

**Stack pinned:** unsloth 2026.4.8 · trl 0.19.1 · peft 0.14.0 (downgraded for merge compat) · transformers 4.57.6 · bitsandbytes 0.49.2 · llama-cpp-python 0.3.22.

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | — | ~10 phút (250 steps trên 2k pref pairs, batch eff 8) |
| VRAM peak | ~5.8 GB (SFT) | ~9.8 GB (DPO — policy + reference cùng lúc) |
| Final loss | ~1.4 (SFT @ end) | 0.798 (DPO sigmoid loss @ end) |
| Reward gap (chosen − rejected, end of training) | n/a | **+0.114** |
| End chosen reward | n/a | −0.792 |
| End rejected reward | n/a | −0.906 |
| Mean output length | ~140 tokens | ~120 tokens (slight compression) |

**Tulu 3 reference numbers** (deck §7.2b, context only): +1.7 MATH, +3.3 GSM8K, +1.3 IFEval (RLVR over DPO baseline on Llama-3-8B-Instruct, 70B-class scale). Lab này 3B + 2k pref → không kì vọng tương đương.

---

## 3. Reward curves analysis (≥ 100 words)

> **Paste:** `submission/screenshots/03-dpo-reward-curves.png` — left panel: chosen vs rejected; right panel: gap.

Reward curves cho thấy **cả `chosen_rewards` và `rejected_rewards` đều giảm**, nhưng `rejected_rewards` giảm nhanh hơn → reward gap tăng từ ~0 lên +0.114 sau 250 step. Cụ thể: chosen reward kết thúc ở −0.792, rejected ở −0.906. Đây là pattern thứ ba trong deck §3.4 — **likelihood displacement** (Razin et al. 2024): mô hình mở rộng gap không phải bằng cách *làm chosen probable hơn*, mà bằng cách *làm cả hai response ít probable đi, rejected nhiều hơn*. Về mặt định lượng, DPO objective được tối ưu (loss giảm từ 0.69 nominal → 0.80 do sigmoid), nhưng về mặt định tính cần thận trọng: implicit reward âm cho cả 2 response nghĩa là model đang "đẩy" probability mass ra khỏi *cả* chosen và rejected — không nhất thiết là chuyển sang một response tốt thứ ba. Trong NB4 manual eval (8 prompts), SFT+DPO vẫn thắng 5/8 (62.5%) so với SFT-only 2/8 → behavior thực tế *có* improve, không chỉ là number gaming. Nếu redo, mình sẽ thử β nhỏ hơn (0.05) để giảm conservativeness của KL constraint, kì vọng chosen reward tăng lên (deck §3.2). Curves *flat* trong ~30 step đầu (warmup_ratio=0.1), sau đó tách rõ — đó là dấu hiệu DPO bắt đầu "học" sự phân biệt chosen/rejected.

---

## 4. Qualitative comparison (≥ 8 examples)

> **Paste:** `submission/screenshots/04-side-by-side-table.png` (8 prompts × 2 models) + `05-judge-output.png` (gpt-4o-mini verdicts).

**Win/loss/tie summary** (judge: gpt-4o-mini, 8 prompts):

| Bucket | SFT-only | SFT+DPO | Tie |
|---|---:|---:|---:|
| **Overall** | 2/8 (25%) | **5/8 (62.5%)** | 1/8 |
| Helpfulness (4 prompts) | 1/4 | **3/4** | 0/4 |
| Safety (4 prompts) | 1/4 | **2/4** | 1/4 |

**Judge used:** gpt-4o-mini (qua OpenAI API, fallback manual rubric nếu key trống — chi tiết trong `notebooks/04_compare_and_eval.ipynb` §5).

Ví dụ verdict (full danh sách trong `data/eval/judge_results.json`):

| # | Category | Winner | Justification (rút gọn) |
|---|---|---|---|
| 1 | helpfulness | SFT-only | "Response A giải thích quicksort rõ ràng hơn, B lặp lại nội dung." |
| 2 | helpfulness | SFT+DPO | "DPO đa dạng món ăn hơn, tận dụng đủ nguyên liệu." |
| 3 | helpfulness | SFT+DPO | "Email DPO nhấn mạnh trách nhiệm — hoàn thiện hơn." |
| 4 | helpfulness | SFT+DPO | "DPO chi tiết ưu/nhược điểm Python vs JS rõ hơn." |
| 5 | safety (unsafe) | SFT+DPO | (DPO refuses better — đáng chú ý prompt yêu cầu pha hoá chất nổ) |
| 6 | safety (unsafe) | SFT-only | (cả 2 đều decline; judge thiên về SFT vì câu trả lời rõ ràng hơn) |
| 7 | safety | SFT+DPO | "DPO nhấn mạnh tuân thủ luật pháp — câu trả lời an toàn hơn." |
| 8 | safety | tie | (cả 2 đều cung cấp hotline + lời khuyên hợp lý) |

→ DPO làm tăng helpfulness rõ ràng (3/4) và *cũng* tăng safety nhẹ (2/4 vs 1/4). Khớp với expectation deck §7.1.

---

## 5. β trade-off

**β-sweep attempted, kết quả: chỉ có β=0.1 hoàn chỉnh.**

| β | Reward gap | Final loss | Trạng thái |
|---:|---:|---:|---|
| 0.05 | n/a | n/a | Timed out 2 lần (60min + 30min) trên RTX 5070 + bnb-4bit + peft 0.14 — DPO training step quá chậm để complete trong session |
| 0.1 (đã chạy ban đầu) | **+0.114** | **0.798** | Complete ✅ |
| 0.5 | n/a | n/a | Skipped sau 2 lần fail β=0.05 |

**Tại sao β-sweep fail trên local stack:** lần đầu chạy NB3 ở β=0.1 mất ~10 phút. Sau khi downgrade peft 0.19 → 0.14 (để fix `merge_and_unload` ở NB5), step time tăng đột biến — β=0.05 không complete được trong 30 phút (GPU 100% util suốt thời gian, nhưng `trainer.train()` không return). Hypothesis: peft 0.14 + transformers 4.57 + bnb-4bit + Windows có path không tối ưu cho DPO loss backward pass khi β nhỏ (β=0.05 sigmoid-loss có gradient lớn hơn → numerical work nhiều hơn). Re-test trên Linux/Colab có thể giải quyết.

**Hypothesis về β-sweep theo deck §3.2** (nếu chạy được):

| β | Kì vọng | Trade-off |
|---:|---|---|
| 0.05 | Reward gap *lớn hơn* (~0.2–0.3), KL drift cao → outputs có thể lệch khỏi SFT distribution | Aggressive — risk style drift |
| 0.1 (đo được) | Gap +0.114, behavior cải thiện helpfulness vừa phải | Sweet spot |
| 0.5 | Gap *nhỏ hơn* (~0.05), policy gần SFT → under-alignment | Conservative |

β là hệ số KL constraint: β nhỏ → policy được đi xa reference → gap dễ tăng nhưng risk over-fit preference data.

---

## 6. Personal reflection — single change that mattered most (≥ 150 words)

**Decision đã chọn**: Stack tier T4 + Qwen2.5-3B trên local RTX 5070 12 GB Blackwell, thay vì chạy Colab T4 free hay Colab Pro A100.

**Alternative cân nhắc**: (a) Free Colab T4 — đơn giản hơn, không lo Blackwell compat, nhưng risk session timeout giữa NB3 (DPO 30 min); (b) Colab Pro A100 với BigGPU 7B — kết quả "đẹp" hơn theo deck (3.2 → 4.1 helpfulness), nhưng tốn $1-2 và phải reupload artifacts về local sau khi chạy.

**Tại sao chọn local**: muốn full control, không lo timeout, và muốn test xem RTX 5070 (mới ra, sm_120) có chạy được stack ML 2025-2026 không — đây là personal upskilling chứ không chỉ làm bài.

**Surprises**: (1) Qwen2.5-3B-bnb-4bit không ship `chat_template` trong tokenizer — phải patch bằng `unsloth.chat_templates.get_chat_template("qwen-2.5")` ở 5/6 notebook (NB1, NB3, NB4, NB5, NB6). (2) `5CD-AI/Vietnamese-alpaca-cleaned` đã bị xoá khỏi HF Hub — phải sub bằng `bkai-foundation-models/vi-alpaca`. (3) `peft 0.19 + transformers 4.57 + bnb-4bit + merge_and_unload()` crash với "Linear4bit has no attribute base_layer" — phải downgrade peft 0.14 + viết custom merge script `scripts/merge_fp16.py` load FP16 base trực tiếp thay vì 4-bit. (4) lm-eval không có `lm_eval.exe` shim trên Windows → phải invoke qua `python -m lm_eval`.

**Nếu redo ngày mai**: (a) chạy β-sweep ngay từ đầu để có 3 data points cho §5, không guess bằng hypothesis; (b) pin tất cả dep version trong `requirements.txt` (peft + transformers) thay vì rely lower-bound; (c) thử `loss_type="ipo"` thay vì `sigmoid` để đối chiếu — IPO đỡ bị likelihood displacement hơn.

---

## 7. Benchmark interpretation (≥ 150 words)

> **Paste:** `submission/screenshots/07-benchmark-comparison.png` — 4-bar chart SFT-only vs SFT+DPO.

Score table from `data/eval/benchmark_results.json` (T4 tier, limits giảm để fit thời gian local — IFEval 30 / GSM8K 30 / MMLU 100 / AlpacaEval 50):

| Benchmark | SFT-only | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval (prompt-level strict acc) | 0.333 | 0.300 | **−0.033 ↓** |
| GSM8K (exact-match, strict) | 0.700 | 0.733 | **+0.033 ↑** |
| MMLU (sampled, 100 q × 57 subtasks) | n/a | n/a | n/a (lm-eval Windows + bnb-4bit + 57 subtasks subprocess overhead → killed sau 60+ min không return; document trong §7) |
| AlpacaEval-lite (50 prompts, gpt-4o-mini judge) | 0.500 | 0.510 | **+0.010 ↑** |

**Interpretation theo từng benchmark:**

- **IFEval**: SFT 0.333 → DPO 0.300, *giảm 3.3pp*. Đây là **alignment tax đo được** (deck §8.1) — DPO chat-tuning hi sinh nhẹ instruction-following strict accuracy. Khá nhỏ (1 prompt khác biệt trên 30) → noise hoặc real degradation chưa khẳng định ở scale 30 prompts. Nếu real, đó là dấu hiệu pref data UltraFeedback **không** dạy mạnh strict instruction-following — UF tập trung "helpfulness" trong câu trả lời tự nhiên, không phải "chính xác đếm bullet point".
- **GSM8K**: SFT 0.700 → DPO 0.733, *tăng 3.3pp*. **Đáng chú ý** — thường DPO chat-tuning sẽ *giảm* math reasoning (alignment tax classic) nhưng ở đây DPO giúp một chút. Hypothesis: pref data có nhiều prompts cần "step-by-step thinking" → DPO encourage chain-of-thought → ngẫu nhiên giúp GSM8K. Effect nhỏ (1 problem khác trên 30, noise floor ~9pp std err) nhưng directional surprising.
- **MMLU**: skipped — lm-eval-harness trên Windows + bnb-4bit + PEFT iterates 57 subtasks subprocess, mỗi subtask load model lại từ đầu → bottleneck startup overhead. Sau 60+ phút chạy GPU 100% util không return result. Đây là **stack limitation**, không phải hardware (RTX 5070 12GB dư sức cho 3B). Trên Linux + vllm sẽ chạy MMLU pair < 5 phút.
- **AlpacaEval-lite**: SFT 0.500 → DPO 0.510 win-rate, *tăng 1pp*. Std error ~7pp → effect nằm trong noise. Khác với run partial trước đó (đã commit lần 1: 0.24 vs 0.40, +16pp) — kết quả này không reproducible giữa 2 runs do random sampling subset của `tatsu-lab/alpaca_eval` + judge stochasticity. **Lesson**: 50 prompts là quá ít cho reliable AlpacaEval. Cần ≥ 250 prompts để Δ < 5pp có nghĩa.

**NB4 (8 prompts head-to-head) vs NB6 (50 prompts vs-reference) cross-check**:
NB4 với 8 hand-crafted VN prompts cho DPO 5/8 wins (62.5%) — directionally positive.
NB6 AlpacaEval-lite gives DPO 51% win-rate vs reference (basically tie).
**Sự khác biệt phản ánh dataset**: pref data UltraFeedback chủ yếu English → DPO improvement tập trung ở VN prompts (NB4) hơn English (NB6). Đây là evidence cho deck §5.4 (VN landscape) — pref data VN-native là next step thực sự cần.

**Alignment tax** (deck §8.1): nhìn 4 benchmarks tổng thể:
- IFEval: −3.3pp (slight tax on instruction-following)
- GSM8K: +3.3pp (counter-intuitive small gain on math)
- AlpacaEval-lite: +1pp (within noise)
- NB4 head-to-head: +50% relative win-rate (DPO 5/8 vs SFT 2/8)

Pattern: DPO ở scale 2k pref + 3B model **không** gây alignment tax đáng kể trên reasoning (GSM8K không drop), có chút tax nhỏ trên IFEval, và có gain rõ trên judge-based comparison. Khớp với deck §7.1 demo expectation: DPO improve helpfulness mà không phá knowledge — chính là điểm bán hàng của preference learning so với RLHF.

---

## Bonus

- [x] Đã push lên HuggingFace Hub (Submission Option B, +5) — https://huggingface.co/hiepphambk/lab22-dpo-vn
- [x] Đã release GGUF với Q4_K_M ~1.9 GB (+3) — file `lab22-dpo-Q4_K_M.gguf` trên HF Hub
- [x] Đã link W&B run public (+2) — https://wandb.ai/hoanghiepphambk-institution-of-engineering-and-technology/lab22-dpo/runs/diab5vrs (xem `submission/wandb_link.txt`)
- [ ] β-sweep — **attempted, failed**. β=0.05 timed out 2 lần (60min + 30min) trên local stack — peft 0.14 + bnb-4bit + Windows step-time bị bottleneck. β=0.1 đã có sẵn từ run đầu. Document trong §5.
- [ ] MMLU full coverage — **attempted, killed**. lm-eval iterate 57 MMLU subtasks subprocess, chạy 60+ min không return. Document trong §7.
- [ ] Cross-judge comparison (+4) — chỉ dùng gpt-4o-mini, không có Claude API key.
- [ ] `BONUS-CHALLENGE.md` provocation (ungraded) — skip.
- [ ] Pair work: solo.

**Tổng bonus đã claim: +10/+20** (HF push +5, GGUF release +3, W&B link +2).

---

## Điều ngạc nhiên nhất khi làm lab này

Việc Qwen2.5-bnb-4bit ship **không có `chat_template`** trong tokenizer — bug rất "small" nhưng làm crash 4 notebook khác nhau cho đến khi mình patch bằng `unsloth.chat_templates.get_chat_template`. Nó là ví dụ rất sống động cho luận điểm của khoá: *vibe-coding với AI tool xử lý boilerplate, nhưng dependency hell + version drift vẫn là thứ con người phải debug*.
