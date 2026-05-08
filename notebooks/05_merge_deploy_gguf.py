# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # NB5 — Merge + Deploy + GGUF
#
# **Stack:** PEFT `merge_and_unload` (FP16 base, not bnb-4bit) + llama.cpp's
# `convert_hf_to_gguf.py` + `llama-quantize.exe` Q4_K_M + llama-cpp-python smoke.
# Maps to deck §7.1 lab brief.
#
# > **Mục tiêu:** export the SFT+DPO adapter as a deployable GGUF Q4_K_M file
# > (~1.8 GB on 3B), then smoke-test it through llama-cpp-python.
# >
# > **Workaround note:** Unsloth's `save_pretrained_merged(method='merged_16bit')`
# > crashes on transformers 4.57+ + peft 0.14+ with bnb 4-bit base
# > (`Linear4bit has no attribute 'base_layer'`). We bypass it by:
# >   1. Loading the *non-quantized* base in FP16 via `transformers.AutoModelForCausalLM`
# >   2. Applying SFT then DPO LoRA on top, calling `merge_and_unload()` after each
# >   3. Saving the merged HF directory
# >   4. Converting via the upstream `convert_hf_to_gguf.py` from llama.cpp
# >   5. Quantizing via `llama-quantize.exe Q4_K_M`
# > These steps run as `scripts/merge_fp16.py` + `tools/` binaries before this NB.

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import subprocess
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()
BASE_MODEL = (
    "unsloth/Qwen2.5-3B-bnb-4bit" if COMPUTE_TIER == "T4"
    else "unsloth/Qwen2.5-7B-bnb-4bit"
)
MERGE_BASE = "Qwen/Qwen2.5-3B" if COMPUTE_TIER == "T4" else "Qwen/Qwen2.5-7B"
MAX_LEN = 512 if COMPUTE_TIER == "T4" else 1024

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DPO_PATH = REPO_ROOT / "adapters" / "dpo"
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
MERGED_PATH = REPO_ROOT / "adapters" / "merged-fp16"
GGUF_DIR = REPO_ROOT / "gguf"
TOOLS_DIR = REPO_ROOT / "tools"
GGUF_DIR.mkdir(parents=True, exist_ok=True)
MERGED_PATH.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), "NB3 must run first"
assert SFT_PATH.exists(), "NB1 must run first"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"MERGE_BASE:      {MERGE_BASE} (FP16 base used for merge — non-bnb)")
print(f"DPO adapter:     {DPO_PATH}")
print(f"merged output:   {MERGED_PATH}")
print(f"GGUF output:     {GGUF_DIR}")

# %%
import torch

assert torch.cuda.is_available()

# %% [markdown]
# ## 1. Merge LoRA adapters into FP16 base
#
# We use `scripts/merge_fp16.py` (a thin wrapper around AutoModel + PEFT
# `merge_and_unload`). It loads `Qwen/Qwen2.5-3B` in BF16 directly, applies SFT
# then DPO LoRA, and saves the merged HF directory.

# %%
merged_marker = MERGED_PATH / "config.json"
if merged_marker.exists() and (MERGED_PATH / "model.safetensors.index.json").exists():
    print(f"Merged FP16 already exists at {MERGED_PATH} — skipping merge step.")
else:
    print("Running scripts/merge_fp16.py …")
    rc = subprocess.run(
        [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
         str(REPO_ROOT / "scripts" / "merge_fp16.py"),
         "--base", MERGE_BASE,
         "--sft", str(SFT_PATH),
         "--dpo", str(DPO_PATH),
         "--out", str(MERGED_PATH)],
        check=True,
    )
    print(f"merge_fp16.py exited with code {rc.returncode}")

print("\nMerged FP16 directory contents:")
for p in sorted(MERGED_PATH.iterdir()):
    if p.suffix in (".safetensors", ".json"):
        print(f"  {p.name:50s}  {p.stat().st_size / 1e6:>8.1f} MB")

# %% [markdown]
# ## 2. Convert merged HF → GGUF FP16 → quantize Q4_K_M
#
# Two-step using llama.cpp upstream tools (downloaded into `tools/`):
#   - `convert_hf_to_gguf.py` writes intermediate FP16 GGUF
#   - `llama-quantize.exe` produces the final Q4_K_M (~1.8 GB)

# %%
gguf_q4 = GGUF_DIR / "lab22-dpo-Q4_K_M.gguf"
gguf_f16 = GGUF_DIR / "lab22-dpo-f16.gguf"

if gguf_q4.exists():
    print(f"Q4_K_M GGUF already exists ({gguf_q4.stat().st_size / 1e9:.2f} GB) — skipping conversion.")
else:
    convert_script = TOOLS_DIR / "convert_hf_to_gguf.py"
    quantize_exe = TOOLS_DIR / "llama-cpp" / "llama-quantize.exe"
    assert convert_script.exists(), f"Missing {convert_script}"
    assert quantize_exe.exists(), f"Missing {quantize_exe}"

    print("Step 2a: HF → FP16 GGUF …")
    subprocess.run(
        [str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
         str(convert_script), str(MERGED_PATH),
         "--outfile", str(gguf_f16),
         "--outtype", "f16"],
        check=True,
    )

    print("\nStep 2b: FP16 → Q4_K_M quantization …")
    subprocess.run(
        [str(quantize_exe), str(gguf_f16), str(gguf_q4), "Q4_K_M"],
        check=True,
    )
    # Free disk: only keep Q4_K_M
    if gguf_f16.exists():
        gguf_f16.unlink()

# %%
print("GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        print(f"  {p.name:40s}  {p.stat().st_size / 1e6:>8.1f} MB")

# %% [markdown]
# ## 3. Smoke test with llama-cpp-python
#
# Loads the Q4_K_M GGUF and runs a Vietnamese smoke prompt through `llama-cpp-python`.
# This is the deliverable for the rubric's "GGUF smoke screenshot" check.

# %%
from llama_cpp import Llama

print(f"Loading: {gguf_q4.name}  ({gguf_q4.stat().st_size / 1e9:.2f} GB)")
llm = Llama(
    model_path=str(gguf_q4),
    n_ctx=MAX_LEN,
    n_gpu_layers=0,           # CPU only — llama-cpp-python wheel is CPU on Windows
    verbose=False,
)
print("Loaded.")

# %% [markdown]
# ### 3a. Smoke prompt + response (deliverable: `06-gguf-smoke.png`)

# %%
SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n  {SMOKE_PROMPT}\n")
print(f"RESPONSE (Q4_K_M GGUF, llama-cpp-python):\n  {response['choices'][0]['message']['content']}")
print(f"\nTokens used: {response['usage']}")

# %% [markdown]
# ## 4. Save deployment metadata

# %%
deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "merge_base": MERGE_BASE,
    "merged_path": str(MERGED_PATH),
    "gguf_path": str(gguf_q4),
    "gguf_size_mb": round(gguf_q4.stat().st_size / 1e6, 1),
    "quantization": "Q4_K_M",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}
(REPO_ROOT / "data" / "eval").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "eval" / "deploy_meta.json").write_text(
    json.dumps(deploy_meta, ensure_ascii=False, indent=2)
)
print(f"Saved data/eval/deploy_meta.json")

# %% [markdown]
# ## 5. Submission checklist
#
# - `make verify` — gatekeeper sẽ list missing artifacts.
# - `submission/screenshots/06-gguf-smoke.png` — chụp output cell 3a (PROMPT + RESPONSE + filename `Q4_K_M.gguf`).
# - `submission/REFLECTION.md` — fill §3 (reward curves) + §6 (key decision) + §7 (benchmark).
