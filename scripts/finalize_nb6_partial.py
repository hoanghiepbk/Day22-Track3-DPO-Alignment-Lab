"""Build benchmark_results.json + 4-bar plot from partial NB6 outputs.

NB6 was killed mid-run for early submission. We have:
  - lm-dpo-ifeval (results JSON written)
  - lm-sft-gsm8k (results JSON written)
  - alpaca_lite_judgments.json (full)

For SFT-IFEval / DPO-GSM8K / MMLU pair: use placeholder None → bar chart will show
just the available pairs. Re-running NB6 later will overwrite this file with full data.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "data" / "eval"


def load_first(pattern: str) -> dict | None:
    matches = sorted(glob.glob(str(EVAL / pattern), recursive=True))
    if not matches:
        return None
    return json.loads(Path(matches[-1]).read_text(encoding="utf-8"))


# ─── Pull existing results ────────────────────────────────────────────
res_dpo_ifeval = load_first("lm-dpo-ifeval/**/results*.json")
res_sft_gsm8k = load_first("lm-sft-gsm8k/**/results*.json")

ifeval_dpo = (
    res_dpo_ifeval["results"]["ifeval"]["prompt_level_strict_acc,none"]
    if res_dpo_ifeval else None
)
gsm8k_sft = (
    res_sft_gsm8k["results"]["gsm8k"]["exact_match,strict-match"]
    if res_sft_gsm8k else None
)

alpaca = load_first("alpaca_lite_judgments.json")
if alpaca:
    sft_wins = sum(1 for j in alpaca if j.get("winner") == "A")
    dpo_wins = sum(1 for j in alpaca if j.get("winner") == "B")
    n = len(alpaca)
    alpaca_sft = sft_wins / n if n else None
    alpaca_dpo = dpo_wins / n if n else None
else:
    alpaca_sft = alpaca_dpo = None

results = {
    "compute_tier": "T4",
    "limits": {"ifeval": 30, "gsm8k": 30, "mmlu": 100, "alpaca_lite": 50},
    "note": (
        "Partial run — NB6 was interrupted for early submission. "
        "IFEval-SFT, GSM8K-DPO, MMLU pairs missing. Re-run NB6 to complete."
    ),
    "metrics": {
        "IFEval (prompt-strict)": {"sft": None, "dpo": ifeval_dpo},
        "GSM8K (exact-match strict)": {"sft": gsm8k_sft, "dpo": None},
        "MMLU (acc, sampled)": {"sft": None, "dpo": None},
        "AlpacaEval-lite (win-rate)": {"sft": alpaca_sft, "dpo": alpaca_dpo},
    },
    "deltas": {
        "IFEval (prompt-strict)": None,
        "GSM8K (exact-match strict)": None,
        "MMLU (acc, sampled)": None,
        "AlpacaEval-lite (win-rate)": (
            (alpaca_dpo - alpaca_sft) if (alpaca_sft is not None and alpaca_dpo is not None) else None
        ),
    },
    "raw": {
        "ifeval-dpo": res_dpo_ifeval["results"] if res_dpo_ifeval else None,
        "gsm8k-sft": res_sft_gsm8k["results"] if res_sft_gsm8k else None,
    },
}

(EVAL / "benchmark_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Wrote data/eval/benchmark_results.json")
print(json.dumps(results["metrics"], ensure_ascii=False, indent=2))


# ─── Plot 4-bar chart with available data ─────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

labels = list(results["metrics"].keys())
sft_vals = [results["metrics"][k]["sft"] for k in labels]
dpo_vals = [results["metrics"][k]["dpo"] for k in labels]

# Replace None with 0 for plotting + annotate "missing"
sft_plot = [v if v is not None else 0 for v in sft_vals]
dpo_plot = [v if v is not None else 0 for v in dpo_vals]

x = np.arange(len(labels))
w = 0.36
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x - w/2, sft_plot, w, label="SFT-only", color="#6c8ebf")
b2 = ax.bar(x + w/2, dpo_plot, w, label="SFT+DPO", color="#c83538")

for i, (sv, dv) in enumerate(zip(sft_vals, dpo_vals)):
    if sv is None:
        ax.text(i - w/2, 0.01, "n/a", ha="center", color="#888", fontsize=9, rotation=90)
    else:
        ax.text(i - w/2, sv + 0.01, f"{sv:.2f}", ha="center", fontsize=9)
    if dv is None:
        ax.text(i + w/2, 0.01, "n/a", ha="center", color="#888", fontsize=9, rotation=90)
    else:
        ax.text(i + w/2, dv + 0.01, f"{dv:.2f}", ha="center", fontsize=9)
    if sv is not None and dv is not None:
        delta = dv - sv
        ax.text(i, max(sv, dv) + 0.06, f"Δ={delta:+.3f}", ha="center", fontsize=10,
                fontweight="bold", color=("#137333" if delta > 0 else "#a50e0e"))

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Score")
ax.set_title("NB6 — Benchmark comparison: SFT-only vs SFT+DPO\n(partial — NB6 interrupted; re-run to fill MMLU + missing pairs)")
ax.set_ylim(0, max(0.9, max(sft_plot + dpo_plot) * 1.15 if any(sft_plot + dpo_plot) else 1.0))
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()

screenshot_dir = REPO / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "07-benchmark-comparison.png", dpi=120, bbox_inches="tight")
print("Saved submission/screenshots/07-benchmark-comparison.png")
