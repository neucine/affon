#!/usr/bin/env python3
"""Rerun a declared cell batch with numerical probes and matched peer timings."""
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from case import contract_hash, normalize


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / ("test/parity/reference" if path in {"probe-capabilities.py", "benchmark-cells-pytorch.py"} else "tools") / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target_for(cases):
    import torch
    backends = {case["id"].split("/")[0] for case in cases}
    if len(backends) != 1:
        raise ValueError("one backend per batch is required")
    backend = backends.pop()
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA unavailable")
        return f"{torch.cuda.get_device_name()}/{platform.release()}/{platform.machine()}"
    if backend == "metal":
        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS unavailable")
    model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    return f"{model}/{platform.mac_ver()[0]}/{platform.machine()}"


def run(args):
    if args.iterations < 1 or args.samples < 3 or args.rounds < 2:
        raise ValueError("need positive iterations, at least three samples and two rounds")
    if not args.binary.is_file():
        raise FileNotFoundError(args.binary)
    catalog_bytes = args.catalog.read_bytes()
    source_catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
    catalog = json.loads(catalog_bytes)
    probes = load("capability_probes", "probe-capabilities.py")
    cases = probes.validate_catalog(catalog)
    selected = set(getattr(args, "cell", ()) or ())
    if selected:
        missing = selected - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"cells absent from catalog: {sorted(missing)}")
        cases = [case for case in cases if case["id"] in selected]
        catalog = {**catalog, "cases": cases}
        catalog_bytes = (json.dumps(catalog, indent=2) + "\n").encode()
    if not cases:
        raise ValueError("empty batch catalog")
    axes = {case['id']: normalize(case) for case in cases}
    if any(value['scope'] != 'operation' for value in axes.values()):
        raise ValueError('this runner accepts operation cases; Program and model piece runners need their own contracts')
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"choose a fresh output directory: {output}")
    target = target_for(cases)
    output.mkdir(parents=True)
    snapshot = output / "catalog.json"
    snapshot.write_bytes(catalog_bytes)
    probe_path = output / "capability.json"
    subprocess.run([sys.executable, str(ROOT / "test/parity/reference/probe-capabilities.py"),
                    "--binary", str(args.binary.resolve()), "--catalog", str(snapshot),
                    "--output", str(probe_path)], check=True)
    probe = json.loads(probe_path.read_text())
    expected_ids = [case["id"] for case in cases]
    if [row["id"] for row in probe["cases"]] != expected_ids:
        raise ValueError("probe omitted or reordered cells")
    importer = load("coverage_importer", "import-coverage-evidence.py")
    outcomes = importer.import_probe(probe_path, set(expected_ids))
    benchmark = load("cell_benchmark", "benchmark-cells-pytorch.py")
    case_by_id = {case['id']: case for case in cases}
    rows = {row["id"]: {"id": row["id"], "axes": axes[row["id"]],
                        "case_sha256": contract_hash(case_by_id[row['id']]),
                        "probe_status": row["status"],
                        "probe_artifact": "capability.json",
                        **({"probe_maximum_error": row["maximum_error"],
                            "probe_tolerance": row["tolerance"]}
                           if "maximum_error" in row else {}),
                        **({"probe_error": row["stderr"]} if "stderr" in row else {}),
                        "operator_peer": "not_run", "program_peer": "not_run"}
            for row in probe["cases"]}
    for kind in ("operator_peer", "program_peer"):
        passed_dir = output / kind
        failed_dir = output / f"{kind}_failures"
        passed_dir.mkdir()
        failed_dir.mkdir()
        for case in cases:
            cell = case["id"]
            if cell not in outcomes:
                rows[cell][kind] = "probe_not_correct"
                continue
            filename = cell.replace("/", "__") + ".json"
            try:
                report = benchmark.one(args.binary, case, args.iterations,
                                       args.samples, args.rounds, target, kind)
            except Exception as exc:
                path = failed_dir / filename
                path.write_text(json.dumps({"cell_id": cell, "kind": kind,
                                            "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
                rows[cell][kind] = "error"
                rows[cell][kind + "_artifact"] = str(path.relative_to(output))
            else:
                passed = report["correctness"]["passed"]
                path = (passed_dir if passed else failed_dir) / filename
                path.write_text(json.dumps(report, indent=2) + "\n")
                if passed:
                    importer.import_benchmark(path, outcomes)
                rows[cell][kind] = "passed" if passed else "incorrect"
                rows[cell][kind + "_artifact"] = str(path.relative_to(output))
                rows[cell][kind + "_affon_ms"] = statistics.median(report["affon_samples_ms"])
                rows[cell][kind + "_peer_ms"] = statistics.median(report["peer_samples_ms"])
                rows[cell][kind + "_ratio"] = report["ratio_of_medians"]
                rows[cell][kind + "_maximum_error"] = report["correctness"]["max_abs_error"]
                rows[cell][kind + "_tolerance"] = report["correctness"]["tolerance"]
            print(f"{cell} {kind}: {rows[cell][kind]}", flush=True)
    for value in outcomes.values():
        value.pop("probe_contract")
        value.pop("peer_context", None)
    (output / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    summary = {"schema_version": 1, "suite": "affon_coverage_batch_run",
               "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
               "source_catalog_sha256": source_catalog_sha256,
               "binary_sha256": probe["binary_sha256"], "target": target,
               "iterations": args.iterations, "samples": args.samples, "rounds": args.rounds,
               "cells": [rows[cell] for cell in expected_ids]}
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    subprocess.run([sys.executable, str(ROOT / "test/parity/report.py"),
                    str(output / "manifest.json"), "--output", str(output / "report.md")], check=True)
    subprocess.run([sys.executable, str(ROOT / "test/parity/index.py")], check=True)
    print(json.dumps({"output": str(output), "candidate_cells": len(cases),
                      "correct": len(outcomes),
                      "operator_passed": sum(row["operator_peer"] == "passed" for row in rows.values()),
                      "program_passed": sum(row["program_peer"] == "passed" for row in rows.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=[],
                        help="run only this exact catalog cell; repeat for a small selection")
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
