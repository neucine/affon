#!/usr/bin/env python3
"""Write the final Apertus prompt logit row as f32 bytes for parity review."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROMPT = [1, 1784, 8961, 1307, 32265, 1395]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cached", action="store_true",
                        help="write the first cached decode step after the prompt")
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA required")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, attn_implementation=args.attention).eval()
    with torch.inference_mode():
        result = model(input_ids=torch.tensor([PROMPT], device="cuda"),
                       use_cache=args.cached)
        first = int(result.logits[:, -1, :].argmax())
        if args.cached:
            if first != 10518:
                raise ValueError(f"unexpected first token {first}")
            result = model(input_ids=torch.tensor([[first]], device="cuda"),
                           past_key_values=result.past_key_values, use_cache=True)
        logits = result.logits[:, -1, :]
    values = logits.float().cpu().numpy().astype("<f4")
    args.output.write_bytes(values.tobytes())
    print(json.dumps({"shape": list(values.shape), "argmax": int(values.argmax()),
                      "prompt": PROMPT, "cached": args.cached,
                      "attention": args.attention,
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
