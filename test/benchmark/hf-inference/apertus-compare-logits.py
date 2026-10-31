#!/usr/bin/env python3
"""Compare every BF16 Affon logit against the PyTorch f32 logit row."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affon-bytes", type=Path, required=True,
                        help="JSON array from apertus-logits.ts")
    parser.add_argument("--pytorch-f32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["full", "cached"], default="full")
    args = parser.parse_args()
    source = args.affon_bytes.read_bytes()
    raw = np.array(json.loads(source), dtype=np.uint8)
    if len(raw) % 2:
        raise ValueError("BF16 byte count must be even")
    affon = (raw.view("<u2").astype("<u4") << 16).view("<f4")
    peer_source = args.pytorch_f32.read_bytes()
    pytorch = np.frombuffer(peer_source, dtype="<f4")
    if affon.shape != pytorch.shape or not np.isfinite(affon).all() or not np.isfinite(pytorch).all():
        raise ValueError("logit shapes or finite-value contracts differ")
    error = np.abs(affon - pytorch)
    worst = int(error.argmax())
    top10_affon = np.argsort(affon)[-10:][::-1].tolist()
    top10_pytorch = np.argsort(pytorch)[-10:][::-1].tolist()
    report = {
        "suite": f"apertus_{args.scope}_logit_parity", "count": len(affon),
        "affon_bf16_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "pytorch_f32_sha256": hashlib.sha256(peer_source).hexdigest(),
        "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean()),
        "rms_error": float(np.sqrt(np.mean(error * error))),
        "within_0_5": int((error <= 0.5).sum()),
        "worst_index": worst, "worst_affon": float(affon[worst]),
        "worst_pytorch": float(pytorch[worst]),
        "argmax_affon": int(affon.argmax()), "argmax_pytorch": int(pytorch.argmax()),
        "top10_affon": top10_affon, "top10_pytorch": top10_pytorch,
        "top10_overlap": len(set(top10_affon) & set(top10_pytorch)),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
