#!/usr/bin/env python3
"""Compatibility entry point; canonical implementation lives in test/parity/reference."""
from pathlib import Path
import runpy

_SOURCE = Path(__file__).resolve().parents[1] / "test/parity/reference/benchmark-cells-pytorch.py"
globals().update(runpy.run_path(str(_SOURCE), run_name="__main__" if __name__ == "__main__" else "parity_reference"))
