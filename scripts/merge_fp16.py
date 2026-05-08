"""Workaround NB5 cell 2: merge SFT+DPO LoRA adapters into FP16 base.

Why this exists: Unsloth's `save_pretrained_merged(method='merged_16bit')` calls
into transformers' bnb 4-bit quantizer reload path, which crashes on
peft + transformers 4.57+ with "Linear4bit has no attribute base_layer".

Workaround: load the *non-quantized* base (Qwen/Qwen2.5-3B), apply the LoRA
adapters in FP16, call merge_and_unload(), save HF-format. The resulting
directory is what NB5's GGUF conversion step expects.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B",
                        help="Non-quantized base for FP16 merge (NOT the bnb-4bit variant)")
    parser.add_argument("--sft", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--dpo", default=str(REPO / "adapters" / "dpo"))
    parser.add_argument("--out", default=str(REPO / "adapters" / "merged-fp16"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[merge] Loading base FP16: {args.base}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If SFT adapter saved its own tokenizer (with chat_template), prefer that.
    sft_tokenizer_dir = Path(args.sft)
    if (sft_tokenizer_dir / "tokenizer_config.json").exists():
        print(f"[merge] Reloading tokenizer with chat_template from {sft_tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(sft_tokenizer_dir))

    print(f"[merge] Apply + merge SFT adapter from {args.sft}")
    model = PeftModel.from_pretrained(model, args.sft)
    model = model.merge_and_unload()
    print("[merge] SFT merged.")

    print(f"[merge] Apply + merge DPO adapter from {args.dpo}")
    model = PeftModel.from_pretrained(model, args.dpo)
    model = model.merge_and_unload()
    print("[merge] DPO merged.")

    print(f"[merge] Saving merged FP16 -> {out}")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    print(f"[merge] Done. {out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
