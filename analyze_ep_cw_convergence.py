#!/usr/bin/env python3
"""Convenience wrapper for the EP/CW convergence analysis CLI."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script = Path(__file__).resolve().parent / "assets" / "cli" / "analyze_ep_cw_convergence.py"
    runpy.run_path(str(script), run_name="__main__")
