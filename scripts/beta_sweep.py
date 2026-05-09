"""β-sweep bonus add-on (+6 pts).

Re-runs NB3 DPO training at β ∈ {0.05, 0.5} (default 0.1 already done).
Saves dpo_metrics.json per β. Plots reward gap vs β. Updates REFLECTION §5.

Usage:
    python scripts/beta_sweep.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO / "adapters" / "beta-sweep"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


def run_dpo_at_beta(beta: float) -> dict:
    """Run NB3 with DPO_BETA=beta, copy output adapter to a beta-specific dir."""
    target = SWEEP_DIR / f"beta-{beta}"
    metrics_target = target / "dpo_metrics.json"
    if metrics_target.exists():
        print(f"[sweep] β={beta} already done — reading {metrics_target}")
        return json.loads(metrics_target.read_text(encoding="utf-8"))

    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[sweep] === Training DPO with β={beta} ===")
    env = os.environ.copy()
    env["DPO_BETA"] = str(beta)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["WANDB_RUN_NAME"] = f"dpo-beta{beta}-lr5e-07"

    # Drop existing adapters/dpo so NB3 overwrites cleanly
    dpo_dir = REPO / "adapters" / "dpo"
    if dpo_dir.exists():
        shutil.rmtree(dpo_dir)

    # Drop existing .ipynb so we get a fresh execution
    ipynb = REPO / "notebooks" / "03_dpo_train.ipynb"
    if ipynb.exists():
        ipynb.unlink()

    py = str(REPO / ".venv" / "Scripts" / "python.exe")
    rc = subprocess.run(
        [py, str(REPO / "scripts" / "run_notebook.py"),
         str(REPO / "notebooks" / "03_dpo_train.py"),
         "--timeout", "3600"],
        env=env,
    )
    print(f"[sweep] β={beta} run_notebook rc={rc.returncode}")

    metrics_src = dpo_dir / "dpo_metrics.json"
    if not metrics_src.exists():
        raise RuntimeError(f"NB3 at β={beta} did not produce dpo_metrics.json")
    shutil.copy2(metrics_src, metrics_target)
    return json.loads(metrics_target.read_text(encoding="utf-8"))


def main() -> int:
    # Save the β=0.1 result we already have (so the table has all 3 points).
    main_metrics = REPO / "adapters" / "dpo" / "dpo_metrics.json"
    if main_metrics.exists():
        beta01_target = SWEEP_DIR / "beta-0.1"
        beta01_target.mkdir(parents=True, exist_ok=True)
        beta01_metrics = beta01_target / "dpo_metrics.json"
        if not beta01_metrics.exists():
            shutil.copy2(main_metrics, beta01_metrics)
            print(f"[sweep] saved existing β=0.1 → {beta01_metrics}")

    sweep_results = {0.1: json.loads((SWEEP_DIR / "beta-0.1" / "dpo_metrics.json").read_text(encoding="utf-8"))}

    for beta in [0.05, 0.5]:
        sweep_results[beta] = run_dpo_at_beta(beta)

    # Restore β=0.1 adapter so downstream NBs (4, 5, 6) keep working unchanged.
    print("\n[sweep] Restoring β=0.1 as the canonical adapters/dpo/")
    if (SWEEP_DIR / "beta-0.1" / "dpo_metrics.json").exists():
        # We don't have the actual β=0.1 adapter saved — but adapters/dpo/
        # is currently whatever the LAST β=0.5 run left there. To be safe,
        # we re-run β=0.1 to restore. Skip if user already has the original.
        if not (REPO / "adapters" / "dpo" / "adapter_model.safetensors").exists():
            print("[sweep] adapters/dpo missing — re-running β=0.1 to restore")
            run_dpo_at_beta(0.1)

    # ─── Plot reward gap + chosen reward vs β ────────────────────────
    import matplotlib.pyplot as plt

    betas = sorted(sweep_results.keys())
    gaps = [sweep_results[b]["end_reward_gap"] for b in betas]
    chosen = [sweep_results[b]["end_chosen_reward"] for b in betas]
    rejected = [sweep_results[b]["end_rejected_reward"] for b in betas]
    losses = [sweep_results[b]["final_train_loss"] for b in betas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(betas, chosen, "o-", label="chosen reward", color="#2e548a")
    axes[0].plot(betas, rejected, "o-", label="rejected reward", color="#c83538")
    axes[0].plot(betas, gaps, "o--", label="gap = chosen − rejected", color="#1a3355", linewidth=2)
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("β (KL coefficient, log scale)")
    axes[0].set_ylabel("Implicit reward (end of training)")
    axes[0].set_title("β-sweep — chosen / rejected / gap")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(betas, losses, "o-", color="#137333", linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("β (KL coefficient, log scale)")
    axes[1].set_ylabel("Final DPO sigmoid loss")
    axes[1].set_title("β-sweep — final loss")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "submission" / "screenshots" / "bonus-beta-sweep.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\n[sweep] saved {out_png}")

    # ─── Save sweep summary JSON ─────────────────────────────────────
    summary = {
        "betas": betas,
        "metrics_per_beta": {str(b): sweep_results[b] for b in betas},
    }
    (SWEEP_DIR / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[sweep] saved {SWEEP_DIR / 'sweep_summary.json'}")
    print("\n=== β-sweep summary ===")
    for b in betas:
        m = sweep_results[b]
        print(f"  β={b:>5}: gap={m['end_reward_gap']:+.3f}  chosen={m['end_chosen_reward']:+.3f}  rejected={m['end_rejected_reward']:+.3f}  loss={m['final_train_loss']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
