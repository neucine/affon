#!/usr/bin/env python3
"""Profile resident Apertus greedy decode against the Affon CUDA probe."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROMPT = [1, 1784, 8961, 1307, 32265, 1395]


def generate(model, steps):
    token = torch.tensor([PROMPT], dtype=torch.long, device="cuda")
    past = None
    ids = list(PROMPT)
    times = []
    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = model(input_ids=token, past_key_values=past, use_cache=True,
                           return_dict=True)
            token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = result.past_key_values
            next_id = token.item()
            torch.cuda.synchronize()
            times.append((time.perf_counter_ns() - start) / 1e6)
            ids.append(next_id)
    return {"output": ids, "step_ms": times,
            "cached_tokens_per_second": (steps - 1) * 1000 / sum(times[1:])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    if args.steps < 2 or args.runs < 1:
        raise ValueError("need at least two generated steps and one run")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA required")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation=args.attention).eval()
    runs = [generate(model, args.steps) for _ in range(args.runs)]
    report = {"suite": "apertus_pytorch_cached_decode", "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "dtype": "bf16",
              "attention": args.attention,
              "checkpoint": str(args.checkpoint), "prompt": PROMPT, "runs": runs}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": runs[0]["output"],
                      "cached_tokens_per_second": [run["cached_tokens_per_second"]
                                                   for run in runs]}))


if __name__ == "__main__":
    main()
