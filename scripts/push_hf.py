"""Push DPO adapter + GGUF + model card to HuggingFace Hub.

Submission Option B (+5 bonus pts) + GGUF release add-on (+3 bonus pts).

Reads HF_TOKEN, HF_REPO, HF_USERNAME from .env. Idempotent — safe to re-run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_REPO = os.environ.get("HF_REPO", "").strip()
HF_USERNAME = os.environ.get("HF_USERNAME", "").strip()

if not TOKEN or not HF_REPO:
    print("ERROR: HF_TOKEN or HF_REPO missing in .env", file=sys.stderr)
    sys.exit(1)

api = HfApi(token=TOKEN)

# ─── 1. Create / verify repo ──────────────────────────────────────────
print(f"[hf] Ensuring repo {HF_REPO} exists …")
create_repo(HF_REPO, token=TOKEN, exist_ok=True, private=False, repo_type="model")
print(f"[hf] Repo OK: https://huggingface.co/{HF_REPO}")

# ─── 2. Write model card ──────────────────────────────────────────────
adapter_card = f"""---
language:
- vi
license: apache-2.0
base_model: Qwen/Qwen2.5-3B
tags:
- dpo
- preference-learning
- vietnamese
- qlora
- lora
- unsloth
- trl
datasets:
- bkai-foundation-models/vi-alpaca
- argilla/ultrafeedback-binarized-preferences-cleaned
pipeline_tag: text-generation
---

# Lab 22 — DPO-Aligned Qwen2.5-3B (VN)

Vietnamese-aligned model produced by VinUniversity AICB Day-22 lab (Track 3 — DPO/ORPO Alignment).

**Pipeline:** SFT (1k VN Alpaca) → DPO (2k UltraFeedback, β=0.1, lr=5e-7) on top of `unsloth/Qwen2.5-3B-bnb-4bit`.

## Files

- `adapter_config.json` + `adapter_model.safetensors` — DPO LoRA (rank=16, α=32). Stack on top of the base + SFT-mini adapter.
- `lab22-dpo-Q4_K_M.gguf` — merged + quantized GGUF for llama.cpp / llama-cpp-python deployment (~1.9 GB).

## Quick start

### Inference via llama-cpp-python
```python
from llama_cpp import Llama
llm = Llama(model_path="lab22-dpo-Q4_K_M.gguf", n_ctx=512)
print(llm.create_chat_completion(messages=[{{"role": "user", "content": "Giải thích quicksort."}}])
        ["choices"][0]["message"]["content"])
```

### Inference via transformers + PEFT
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype="bfloat16", device_map="cuda")
model = PeftModel.from_pretrained(base, "{HF_REPO}")  # this DPO adapter
```

## Training details

| Hyperparameter | Value |
|---|---|
| Base | `unsloth/Qwen2.5-3B-bnb-4bit` (4-bit NF4) |
| LoRA rank · α | 16 · 32 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| DPO β | 0.1 |
| Learning rate | 5e-7 |
| Optimizer | adamw_8bit |
| Effective batch | 1 × 8 grad-accum = 8 |
| Epochs | 1 |
| Max seq length | 512 |
| Compute | RTX 5070 12 GB (Blackwell sm_120, CUDA 12.8) |

## Evaluation results

| Metric | SFT-only | SFT+DPO |
|---|---:|---:|
| Reward gap (chosen − rejected, end of training) | n/a | **+0.114** |
| Final DPO loss | n/a | 0.798 |
| AlpacaEval-lite (50 prompts, gpt-4o-mini judge) | 0.50 | 0.47 |
| Manual eval (8 VN prompts, judge gpt-4o-mini) | 2/8 wins | **5/8 wins (62.5%)** |

See full report (incl. reward curves analysis, alignment-tax interpretation, and W&B run link) in the
[lab repo](https://github.com/{HF_USERNAME}/Day22-Track3-DPO-Alignment-Lab).

## Caveats

- Trained on English UltraFeedback pref data — VN behavior improves via transfer; native-VN pref dataset would be better (deck §5.4).
- 3B + 1k SFT + 2k DPO is *demonstrative scale*, not production-ready. For production, use ≥ 7B base + ≥ 50k pref pairs.
- Likelihood displacement observed (deck §3.4): both chosen and rejected reward decrease, gap widens because rejected falls faster.

## Citation / acknowledgements

- Lab template: VinUniversity AICB Day-22 (Track 3, A20 cohort 2026).
- Stack: Unsloth, TRL, PEFT, bitsandbytes, llama.cpp, lm-eval-harness.

Trained by Phạm Hữu Hoàng Hiệp (MSSV 2A202600415).
"""

readme_path = REPO / "adapters" / "dpo" / "README.md"
readme_path.write_text(adapter_card, encoding="utf-8")
print(f"[hf] Wrote model card: {readme_path}")

# ─── 3. Upload DPO adapter folder ─────────────────────────────────────
print(f"[hf] Uploading adapters/dpo/ → {HF_REPO} …")
api.upload_folder(
    folder_path=str(REPO / "adapters" / "dpo"),
    repo_id=HF_REPO,
    repo_type="model",
    path_in_repo="dpo-adapter",
    commit_message="Upload DPO LoRA adapter",
)
print(f"[hf] Adapter uploaded.")

# ─── 4. Upload GGUF Q4_K_M ────────────────────────────────────────────
gguf_path = REPO / "gguf" / "lab22-dpo-Q4_K_M.gguf"
if gguf_path.exists():
    print(f"[hf] Uploading {gguf_path.name} ({gguf_path.stat().st_size / 1e9:.2f} GB) …")
    api.upload_file(
        path_or_fileobj=str(gguf_path),
        path_in_repo=gguf_path.name,
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Upload Q4_K_M GGUF",
    )
    print(f"[hf] GGUF uploaded.")
else:
    print(f"[hf] WARN: {gguf_path} missing — skip GGUF upload.")

# ─── 5. Top-level README on the repo ──────────────────────────────────
api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Update README / model card",
)

print(f"\n[hf] DONE. Visit: https://huggingface.co/{HF_REPO}")
