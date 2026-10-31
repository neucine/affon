#!/usr/bin/env python3
"""Profile resident Apertus greedy decode against the Affon CUDA probe."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, StaticCache
from transformers.cache_utils import StaticLayer

PROMPT = [1, 1784, 8961, 1307, 32265, 1395]


class FunctionalStaticLayer(StaticLayer):
    """Match Affon's padded prefill and full-cache `where` decode update."""

    def update(self, key_states, value_states, *args, **kwargs):
        first = not self.is_initialized
        if first:
            self.lazy_initialization(key_states, value_states)
        length = key_states.shape[-2]
        if first:
            self.keys = torch.cat((key_states, self.keys[:, :, length:]), dim=2)
            self.values = torch.cat((value_states, self.values[:, :, length:]), dim=2)
        else:
            positions = torch.arange(self.max_cache_len, device=self.device)
            mask = (positions == self.cumulative_length).view(1, 1, -1, 1)
            self.keys = torch.where(mask, key_states, self.keys)
            self.values = torch.where(mask, value_states, self.values)
        self.cumulative_length.add_(length)
        return self.keys, self.values


def generate(model, steps, cache_kind, cache_capacity, selection, cache_update):
    token = torch.tensor([PROMPT], dtype=torch.long, device="cuda")
    past = StaticCache(config=model.config, max_cache_len=cache_capacity) if cache_kind == "static" else None
    if cache_update == "functional":
        if any(type(layer) is not StaticLayer for layer in past.layers):
            raise ValueError("functional cache requires ordinary static layers")
        past.layers = [FunctionalStaticLayer(max_cache_len=cache_capacity) for _ in past.layers]
    ids = list(PROMPT)
    times = []
    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = model(input_ids=token, past_key_values=past, use_cache=True,
                           cache_position=(torch.arange(len(PROMPT), device="cuda") if len(ids) == len(PROMPT)
                                           else torch.tensor([len(ids) - 1], device="cuda")) if cache_kind == "static" else None,
                           return_dict=True)
            if selection in ("cpu-full", "cpu-bf16"):
                if selection == "cpu-bf16":
                    bits = result.logits[0, -1].view(torch.uint16).cpu().numpy()
                    row = (bits.astype(np.uint32) << 16).view(np.float32)
                else:
                    row = result.logits[0, -1].float().cpu().numpy()
                if not bool(np.isfinite(row).all()):
                    raise ValueError("nonfinite logits")
                next_id = int(row.argmax())
                token = torch.tensor([[next_id]], dtype=torch.long, device="cuda")
            else:
                token = result.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                next_id = token.item()
            past = result.past_key_values
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
    parser.add_argument("--cache", choices=["dynamic", "static"], default="static")
    parser.add_argument("--cache-capacity", type=int, default=320)
    parser.add_argument("--selection", choices=["gpu-id", "cpu-full", "cpu-bf16"], default="cpu-bf16")
    parser.add_argument("--cache-update", choices=["inplace", "functional"], default="functional")
    args = parser.parse_args()
    if args.steps < 2 or args.runs < 1:
        raise ValueError("need at least two generated steps and one run")
    if args.cache == "static" and args.cache_capacity < len(PROMPT) + args.steps:
        raise ValueError("static cache must hold prompt and generated tokens")
    if args.cache == "dynamic" and args.cache_update == "functional":
        raise ValueError("functional cache updates require a static cache")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA required")
    if not args.checkpoint.is_dir():
        raise FileNotFoundError(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation=args.attention).eval()
    runs = [generate(model, args.steps, args.cache, args.cache_capacity, args.selection,
                     args.cache_update)
            for _ in range(args.runs)]
    report = {"suite": "apertus_pytorch_cached_decode", "torch": torch.__version__,
              "device": torch.cuda.get_device_name(), "dtype": "bf16",
              "attention": args.attention,
              "cache": args.cache, "cache_capacity": args.cache_capacity if args.cache == "static" else None,
              "selection": args.selection,
              "cache_update": args.cache_update if args.cache == "static" else None,
              "checkpoint": str(args.checkpoint), "prompt": PROMPT, "runs": runs}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": runs[0]["output"],
                      "cached_tokens_per_second": [run["cached_tokens_per_second"]
                                                   for run in runs]}))


if __name__ == "__main__":
    main()
