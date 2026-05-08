"""Helper: load .env then execute a jupytext notebook in-place.

Usage:
    python scripts/run_notebook.py notebooks/01_sft_mini.py [--timeout 1800]

Why this script: Makefile is unix-pathed (.venv/bin/python) and notebooks rely on
env vars from .env (SFT_DATASET override, API keys, COMPUTE_TIER). This wrapper:
  1. Loads .env into os.environ
  2. Converts .py -> .ipynb via jupytext (preserves output cells once executed)
  3. Executes notebook in-place via nbclient
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", help="Path to notebook .py (jupytext py:percent format)")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    load_dotenv(repo / ".env")

    nb_py = Path(args.notebook).resolve()
    if not nb_py.exists():
        print(f"ERROR: {nb_py} not found", file=sys.stderr)
        return 1

    nb_ipynb = nb_py.with_suffix(".ipynb")

    # Convert .py -> .ipynb (jupytext --update preserves outputs if already converted)
    import jupytext

    nb = jupytext.read(str(nb_py))
    jupytext.write(nb, str(nb_ipynb))
    print(f"[run_notebook] {nb_py.name} -> {nb_ipynb.name}", flush=True)

    # Execute
    import nbformat
    from nbclient import NotebookClient

    nb_obj = nbformat.read(str(nb_ipynb), as_version=4)
    client = NotebookClient(
        nb_obj,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_py.parent)}},
    )
    client.execute()
    nbformat.write(nb_obj, str(nb_ipynb))
    print(f"[run_notebook] executed -> {nb_ipynb}", flush=True)

    # Sync .ipynb back to .py (so source-of-truth stays current)
    nb_synced = jupytext.read(str(nb_ipynb))
    jupytext.write(nb_synced, str(nb_py))
    return 0


if __name__ == "__main__":
    sys.exit(main())
