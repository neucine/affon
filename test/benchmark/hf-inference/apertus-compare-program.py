#!/usr/bin/env python3
"""Compare cached decode steps and generated IDs from Affon and PyTorch."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def affon_run(path):
    source = path.read_bytes()
    rows = [json.loads(line.removeprefix("APERTUS_PROFILE "))
            for line in source.decode().splitlines() if line.startswith("APERTUS_PROFILE ")]
    if len(rows) != 1:
        raise ValueError(f"expected one APERTUS_PROFILE row in {path}")
    row = rows[0]
    times = row["stepMs"][1:]
    return {"source_sha256": hashlib.sha256(source).hexdigest(),
            "prompt": row["prompt"], "output": row["output"],
            "capacity": row["capacity"], "step_ms": row["stepMs"],
            "cached_tokens_per_second": len(times) * 1000 / sum(times),
            "median_cached_step_ms": statistics.median(times)}


def affon_result(path):
    source = path.read_bytes()
    result = json.loads(source)
    runs = []
    for step_ms in result["optimized_step_ms"]:
        times = step_ms[1:]
        runs.append({"source_sha256": hashlib.sha256(source).hexdigest(),
                     "prompt": result["prompt"], "output": result["output"],
                     "capacity": result["cache_capacity"], "step_ms": step_ms,
                     "cached_tokens_per_second": len(times) * 1000 / sum(times),
                     "median_cached_step_ms": statistics.median(times)})
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affon", type=Path, action="append")
    parser.add_argument("--affon-result", type=Path)
    parser.add_argument("--pytorch", type=Path, required=True)
    parser.add_argument("--pytorch-eager", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.affon) == bool(args.affon_result):
        parser.error("provide Affon run logs or one retained Affon result")
    affon = [affon_run(path) for path in args.affon] if args.affon else affon_result(args.affon_result)
    pytorch = json.loads(args.pytorch.read_text())
    eager = json.loads(args.pytorch_eager.read_text()) if args.pytorch_eager else None
    outputs = [row["output"] for row in affon + pytorch["runs"] +
               (eager["runs"] if eager else [])]
    if any(value != outputs[0] for value in outputs):
        raise ValueError("generated token IDs differ")
    if any(row["prompt"] != affon[0]["prompt"] for row in affon):
        raise ValueError("Affon prompts differ")
    report = {"suite": "apertus_program_peer", "output_ids_match": True,
              "timing_scope": "resident model, cached greedy step with synchronized logits selection",
              "cache_protocol": {"affon": f"fixed capacity {affon[0]['capacity']}, functional full-cache update",
                                 "pytorch": (f"static capacity {pytorch['cache_capacity']}, {pytorch.get('cache_update', 'inplace')} update"
                                             if pytorch.get("cache") == "static" else "dynamic cache")},
              "selection_protocol": {"affon": "full BF16 logit row to CPU, JavaScript argmax",
                                     "pytorch": pytorch.get("selection", "gpu-id")},
              "prompt": affon[0]["prompt"], "output": outputs[0],
              "affon": affon, "pytorch_sdpa": pytorch,
              **({"pytorch_eager": eager} if eager else {})}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"affon_tokens_per_second":
                      [row["cached_tokens_per_second"] for row in affon],
                      "pytorch_tokens_per_second":
                      [row["cached_tokens_per_second"] for row in pytorch["runs"]]}))


if __name__ == "__main__":
    main()
