"""Download a pinned GPT-2 checkpoint and produce independent CPU reference data."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--revision", required=True, help="Hub commit SHA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeded-wide", action="store_true", help="Use deterministic width-64 random weights; tokenizer still comes from pinned model")
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("--revision must be a full lowercase commit SHA")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    torch.set_num_threads(4)
    if args.seeded_wide:
        torch.manual_seed(1729)
        config = GPT2Config(vocab_size=len(tokenizer), n_embd=64, n_head=4, n_layer=2,
                            n_positions=128, bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id, attn_implementation="eager")
        model = GPT2LMHeadModel(config).cpu().eval()
    else:
        model = GPT2LMHeadModel.from_pretrained(
            args.model, revision=args.revision, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager",
        ).cpu().eval()
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    tensors = {}
    cases = []
    # One token, multiple tokens, Unicode/whitespace, and a special token.
    prompts = ["Hello", "The capital of France is", " café\nHello!", "<|endoftext|>Hello"]
    if args.seeded_wide:
        prompts.append("The quick brown fox jumps over the lazy dog. " * 4)
    for index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        current = torch.tensor([ids], dtype=torch.long)
        with torch.inference_mode():
            result = model(current, use_cache=False, output_hidden_states=True)
            tensors[f"case_{index}.logits"] = result.logits.contiguous()
            for layer, hidden in enumerate(result.hidden_states):
                tensors[f"case_{index}.hidden_{layer}"] = hidden.contiguous()
            for _ in range(4):
                logits = model(current, use_cache=False).logits
                next_id = logits[:, -1].argmax(-1).reshape(1, 1)
                current = torch.cat([current, next_id], dim=1)
                if next_id.item() == model.config.eos_token_id:
                    break
        cases.append({"prompt": prompt, "input_ids": ids, "generated_ids": current[0].tolist(),
                      "decoded": tokenizer.decode(current[0], skip_special_tokens=False), "max_new_tokens": 4})
    save_file(tensors, str(output / "reference.safetensors"))
    hashes = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != 'reference.json' and not path.name.startswith('audit-'):
            with path.open('rb') as handle:
                hashes[path.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    manifest = {
        "format": "affon-hf-inference-reference/v1", "model_id": args.model,
        "weight_origin": "seeded-random-width-64-seed-1729" if args.seeded_wide else "pretrained",
        "revision": args.revision, "reference_device": "cpu", "dtype": "float32",
        "versions": {name: importlib.metadata.version(name)
                     for name in ["torch", "transformers", "tokenizers", "safetensors", "huggingface-hub"]},
        "artifact_note": "HF model reserialized as f32 SafeTensors; tokenizer exported as tokenizer.json. No Affon weight-name remapping.",
        "sha256": hashes, "cases": cases,
    }
    (output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(cases)} reference cases to {output}")


if __name__ == "__main__":
    main()
