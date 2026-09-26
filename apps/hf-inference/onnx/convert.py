"""Compatibility entry point; conversion is owned by @affon/onnx."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[3]/'packages/@affon/onnx/tools/convert.py'),run_name='__main__')
