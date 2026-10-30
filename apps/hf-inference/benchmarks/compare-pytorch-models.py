#!/usr/bin/env python3
"""Run resident PyTorch peers for the Affon GPT-2 and ViT benchmark contracts."""
import argparse
import json
import math
from pathlib import Path
import platform
import time


def snapshot(cache, model_id, revision):
    directory = cache / ("models--" + model_id.replace("/", "--")) / revision
    manifest = json.loads((directory / "affon-snapshot.json").read_text())
    if manifest["model_id"] != model_id or manifest["revision"] != revision:
        raise ValueError("snapshot identity mismatch")
    return directory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--affon-run", type=Path, required=True)
    p.add_argument("--normalized-affon", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    a = p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageClassification, AutoTokenizer
    normalized = json.loads(a.normalized_affon.read_text())
    run = json.loads((a.affon_run / "run.json").read_text())
    if normalized["engine"] != "affon":
        raise ValueError("expected normalized Affon report")
    contracts = {case["id"]: case["contract"] for case in normalized["cases"]}
    if len(contracts) != len(normalized["cases"]):
        raise ValueError("duplicate normalized case")
    peer_cases = []
    for item in run["runs"]:
        source = json.loads((a.affon_run / item["report"]).read_text())
        family, backend = source["family"], source["device"]
        metric = "generate_ms" if family == "gpt2" else "forward_with_readback_ms"
        key = f"{family}/{backend}/{metric}"
        if key not in contracts or source["measured_runs"] < 3:
            raise ValueError(f"missing or incomplete Affon contract: {key}")
        if backend not in ("cpu", "metal", "cuda"):
            raise ValueError(f"unsupported peer backend: {backend}")
        if backend == "metal" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; refusing CPU fallback")
        if backend == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU fallback")
        device = {"metal": "mps", "cuda": "cuda", "cpu": "cpu"}[backend]
        def sync():
            if device == "mps": torch.mps.synchronize()
            if device == "cuda": torch.cuda.synchronize()
        folder = snapshot(a.cache, source["model_id"], source["revision"])
        if family == "vit":
            model = AutoModelForImageClassification.from_pretrained(folder, local_files_only=True, dtype=torch.float32).eval().to(device)
            fixture = Path(source["input_fixture"])
            if not fixture.is_absolute():
                fixture = a.affon_run / fixture
            pixels = torch.tensor(json.loads(fixture.read_text()), dtype=torch.float32, device=device)
            expected = torch.tensor(source["reference_logits"], dtype=torch.float32)
            if expected.numel() != model.config.num_labels:
                raise ValueError("Affon ViT reference logits missing or wrong size")

            def consume():
                logits = model(pixel_values=pixels).logits.cpu().flatten()
                return logits

            with torch.inference_mode():
                actual = consume()
                if not torch.allclose(actual, expected, atol=a.atol, rtol=a.rtol):
                    raise ValueError(f"ViT output parity failed: max error {(actual-expected).abs().max().item()}")
                correctness = {"maximum_absolute_logit_error": (actual-expected).abs().max().item(),
                               "atol": a.atol, "rtol": a.rtol}
                samples = []
                for iteration in range(source["measured_runs"] + 1):
                    sync()
                    started = time.perf_counter_ns()
                    consume()
                    sync()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if iteration: samples.append(elapsed)
        elif family == "gpt2":
            model = AutoModelForCausalLM.from_pretrained(folder, local_files_only=True, dtype=torch.float32).eval().to(device)
            tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            prompt = source["workload"]["prompt"]
            budget = source["workload"]["max_new_tokens"]
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            if ids.numel() != source["workload"]["prompt_tokens"]:
                raise ValueError("GPT-2 tokenization differs from Affon")

            def consume():
                sequence = ids
                output_ids = ids[0].tolist()
                for _ in range(budget):
                    logits = model(input_ids=sequence, use_cache=False).logits[:, -1, :]
                    next_id = logits.argmax(dim=-1).item()
                    output_ids.append(next_id)
                    sequence = torch.cat((sequence, torch.tensor([[next_id]], device=device)), dim=1)
                    if next_id == model.config.eos_token_id:
                        break
                return output_ids

            with torch.inference_mode():
                if tokenizer.decode(consume(), skip_special_tokens=True) != source["output"]["text"]:
                    raise ValueError("GPT-2 generated text differs from Affon")
                correctness = {"generated_text_equal": True}
                samples = []
                for iteration in range(source["measured_runs"] + 1):
                    sync()
                    started = time.perf_counter_ns()
                    consume()
                    sync()
                    if iteration: samples.append((time.perf_counter_ns() - started) / 1e6)
        else:
            raise ValueError(f"unsupported family: {family}")
        if len(samples) != source["measured_runs"] or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError("incomplete peer samples")
        peer_cases.append({"id": key, "correct": True, "correctness": correctness,
                           "contract": contracts[key], "samples_ms": samples})
        print(f"{key}: {samples}", flush=True)
        del model
        if device == "mps": torch.mps.empty_cache()
        if device == "cuda": torch.cuda.empty_cache()
    if set(case["id"] for case in peer_cases) != set(contracts):
        raise ValueError("peer case matrix incomplete")
    result = {"schema_version": 1, "suite": "affon_model_performance", "target": normalized["target"],
              "engine": "pytorch", "versions": {"torch": torch.__version__, "platform": platform.platform()},
              "cases": peer_cases}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
