#!/usr/bin/env python3
"""Import exact cell evidence; keep older reports without path contracts diagnostic."""
import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"operator_peer", "program_peer", "self_baseline"}
CONTRACT_FIELDS = {"shape", "dtype", "layout", "timing_scope", "optimization_level", "input_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact(path, contract):
    return {"artifact": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "contract": "sha256:" + digest(contract)}


def import_probe(path, known_ids):
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1 or report.get("suite") != "affon_capability_probe_run":
        raise ValueError(f"{path}: invalid probe report")
    if not report.get("binary_sha256") or not report.get("catalog_sha256"):
        raise ValueError(f"{path}: binary or catalog hash missing")
    outcomes = {}
    for row in report.get("cases", []):
        cell = row.get("id")
        if cell not in known_ids or cell in outcomes:
            raise ValueError(f"{path}: unknown or duplicate cell {cell}")
        if row.get("status") != "correct":
            continue
        if not row.get("selected_path") or not math.isfinite(row.get("maximum_error", float("nan"))) or row["maximum_error"] > row.get("tolerance", -1):
            raise ValueError(f"{path}: invalid correctness evidence for {cell}")
        contract = {"cell": cell, "shape": row["shape"], "path": row["selected_path"],
                    **({"input_shapes": row["input_shapes"]} if row.get("input_shapes") else {}),
                    **({"output_shape": row["output_shape"]} if row.get("output_shape") is not None else {}),
                    **({"output_dtype": row["output_dtype"]} if row.get("output_dtype") else {}),
                    **({"outputs": row["outputs"]} if row.get("outputs") else {}),
                    "compile_options": row.get("compile_options", {"dense_inputs": False}),
                    "binary_sha256": report["binary_sha256"], "catalog_sha256": report["catalog_sha256"]}
        outcomes[cell] = {"support": "supported", "evidence": {"correctness": artifact(path, contract)},
                          "probe_contract": contract}
    return outcomes


def finite_samples(values):
    return isinstance(values, list) and len(values) >= 3 and all(
        isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in values)


def validate_baseline_evaluation(evaluation, path):
    sources = evaluation.get("session_artifacts", [])
    if len(sources) < 2:
        raise ValueError(f"{path}: session evaluation lacks source artifacts")
    paths = [Path(item["path"]) for item in sources]
    if len(set(paths)) != len(paths) or any(not source.is_file() for source in paths):
        raise ValueError(f"{path}: session evaluation source missing or duplicated")
    for source, item in zip(paths, sources):
        if hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"{path}: session evaluation source changed")
    spec = importlib.util.spec_from_file_location("evaluate_cell_baseline", ROOT / "tools/evaluate-cell-baseline.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    reports = [json.loads(source.read_text()) for source in paths]
    limits = evaluation["limits"]
    recomputed = evaluator.evaluate(reports, limits["maximum_ratio"], limits["maximum_span_ms"], paths)
    if recomputed != evaluation:
        raise ValueError(f"{path}: session evaluation differs from source sessions")


def import_benchmark(path, outcomes):
    report = json.loads(path.read_text())
    if report.get("schema_version") != 1 or report.get("suite") != "affon_cell_benchmark":
        raise ValueError(f"{path}: legacy or unsupported benchmark format; diagnostic only")
    cell, kind = report.get("cell_id"), report.get("kind")
    if kind not in KINDS or cell not in outcomes:
        raise ValueError(f"{path}: unknown kind or unprobed cell")
    if not report.get("target") or not report.get("affon_binary_sha256") or not report.get("peer_engine"):
        raise ValueError(f"{path}: target, binary or peer identity missing")
    if report["affon_binary_sha256"] != outcomes[cell]["probe_contract"]["binary_sha256"]:
        raise ValueError(f"{path}: Affon binary differs from capability probe")
    contract = report.get("contract")
    if not isinstance(contract, dict) or CONTRACT_FIELDS - set(contract) or any(not contract[key] for key in CONTRACT_FIELDS - {'shape'}) or not isinstance(contract['shape'], list):
        raise ValueError(f"{path}: incomplete workload contract")
    if contract["dtype"] != cell.split("/")[1] or contract["shape"] != outcomes[cell]["probe_contract"]["shape"]:
        raise ValueError(f"{path}: dtype or shape differs from capability probe")
    probe_input_shapes = outcomes[cell]["probe_contract"].get("input_shapes")
    if probe_input_shapes is not None and contract.get("input_shapes") != probe_input_shapes:
        raise ValueError(f"{path}: base input shapes differ from capability probe")
    probe_output_shape = outcomes[cell]["probe_contract"].get("output_shape")
    if probe_output_shape is not None and contract.get("output_shape") != probe_output_shape:
        raise ValueError(f"{path}: output shape differs from capability probe")
    probe_output_dtype = outcomes[cell]["probe_contract"].get("output_dtype")
    if probe_output_dtype is not None and contract.get("output_dtype") != probe_output_dtype:
        raise ValueError(f"{path}: output dtype differs from capability probe")
    probe_outputs = outcomes[cell]["probe_contract"].get("outputs")
    if probe_outputs is not None and contract.get("outputs") != probe_outputs:
        raise ValueError(f"{path}: output list differs from capability probe")
    if contract.get("compile_options", {"dense_inputs": False}) != outcomes[cell]["probe_contract"]["compile_options"]:
        raise ValueError(f"{path}: compile options differ from capability probe")
    if kind == "program_peer":
        program = contract.get("program")
        if (not isinstance(program, dict) or not isinstance(program.get("id"), str) or not program["id"] or
                not isinstance(program.get("ops"), list) or len(program["ops"]) < 2 or
                program["ops"][0] != cell.split("/")[2] or
                any(not isinstance(op, str) or not op for op in program["ops"])):
            raise ValueError(f"{path}: Program graph contract missing or mismatched")
        if "add_bias" in program["ops"] and not re.fullmatch(r"[0-9a-f]{64}", str(contract.get("extra_input_sha256", ""))):
            raise ValueError(f"{path}: Program bias input hash missing")
    selected_path = report.get("selected_path")
    if kind == "operator_peer" and selected_path != outcomes[cell]["probe_contract"]["path"]:
        raise ValueError(f"{path}: selected implementation differs from capability probe")
    if not isinstance(selected_path, list) or not selected_path:
        raise ValueError(f"{path}: selected path missing")
    if report.get("correctness", {}).get("passed") is not True:
        raise ValueError(f"{path}: correctness not established")
    if not finite_samples(report.get("affon_samples_ms")) or not finite_samples(report.get("peer_samples_ms")):
        raise ValueError(f"{path}: raw repeated samples missing")
    if kind == "self_baseline" and report["peer_engine"] != "affon":
        raise ValueError(f"{path}: self baseline must use Affon")
    evaluation_sha256 = None
    if kind == "self_baseline":
        evaluation_path = Path(report.get("baseline_evaluation", ""))
        if not evaluation_path.is_file():
            raise ValueError(f"{path}: accepted session evaluation missing")
        evaluation = json.loads(evaluation_path.read_text())
        if (evaluation.get("suite") != "affon_cell_baseline_evaluation" or
                evaluation.get("accepted") is not True or
                evaluation.get("cell_id") != cell or
                evaluation.get("target") != report["target"] or
                evaluation.get("binary_sha256") != report["affon_binary_sha256"] or
                evaluation.get("selected_path") != selected_path or
                evaluation.get("contract", {}).get("shape") != contract["shape"] or
                evaluation.get("contract", {}).get("dtype") != contract["dtype"] or
                evaluation.get("contract", {}).get("timing_scope") != contract["timing_scope"]):
            raise ValueError(f"{path}: session evaluation rejected or mismatched")
        validate_baseline_evaluation(evaluation, evaluation_path)
        evaluation_sha256 = hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
    if kind != "self_baseline" and report["peer_engine"] == "affon":
        raise ValueError(f"{path}: external peer required")
    if kind in {"operator_peer", "program_peer"}:
        peer_context = {"target": report["target"], "input_sha256": contract["input_sha256"],
                        "layout": contract["layout"], "dtype": contract["dtype"],
                        "shape": contract["shape"],
                        "compile_options": contract.get("compile_options", {"dense_inputs": False})}
        previous = outcomes[cell].setdefault("peer_context", peer_context)
        if peer_context != previous:
            raise ValueError(f"{path}: peer target or input contract differs for cell")
    if kind in outcomes[cell]["evidence"]:
        raise ValueError(f"{path}: duplicate {kind} evidence for {cell}")
    evidence_contract = {"cell": cell, "target": report["target"],
        "kind": kind, "contract": contract, "selected_path": selected_path,
        "affon_binary_sha256": report["affon_binary_sha256"], "peer_engine": report["peer_engine"]}
    if evaluation_sha256 is not None:
        evidence_contract["baseline_evaluation_sha256"] = evaluation_sha256
    outcomes[cell]["evidence"][kind] = artifact(path, evidence_contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, action="append", required=True)
    parser.add_argument("--benchmark", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("coverage_matrix", ROOT / "tools/coverage-matrix.py")
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    policy = json.loads((ROOT / "test/benchmarks/coverage-policy.json").read_text())
    diagnostics = json.loads((ROOT / "test/benchmarks/workload-coverage.json").read_text())
    known_ids = {row["id"] for row in matrix.build(policy, diagnostics)["cells"]}
    outcomes = {}
    for path in args.probe:
        for cell, value in import_probe(path, known_ids).items():
            if cell in outcomes:
                raise ValueError(f"duplicate probe outcome: {cell}")
            outcomes[cell] = value
    for path in args.benchmark:
        import_benchmark(path, outcomes)
    for value in outcomes.values():
        value.pop("probe_contract")
        value.pop("peer_context", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outcomes, indent=2) + "\n")
    print(json.dumps({"supported_cells": len(outcomes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
