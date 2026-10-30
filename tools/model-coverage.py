#!/usr/bin/env python3
"""Show registered model performance cases, including missing evidence."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def comparator():
    path = Path(__file__).with_name("compare-model-performance.py")
    spec = importlib.util.spec_from_file_location("model_performance_comparator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registry(path):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != 1 or value.get("suite") != "affon_model_coverage" or not value.get("cases"):
        raise ValueError("invalid model coverage registry")
    required = value.get("required_archetypes")
    if not isinstance(required, dict) or not required or any(
        not isinstance(group, str) or not group or not isinstance(names, list) or not names or
        any(not isinstance(name, str) or not name for name in names) or len(names) != len(set(names))
        for group, names in required.items()
    ):
        raise ValueError("invalid required archetypes")
    seen = set()
    for row in value["cases"]:
        key = (row.get("target_group"), row.get("suite_id"), row.get("id"))
        if not all(isinstance(part, str) and part for part in key) or key in seen:
            raise ValueError(f"invalid or duplicate coverage case: {key}")
        seen.add(key)
        if row["target_group"] not in required or row.get("archetype") not in required[row["target_group"]]:
            raise ValueError(f"{key}: unknown workload archetype")
        if not isinstance(row.get("peer"), str) or not row["peer"]:
            raise ValueError(f"{key}: peer engine missing")
        ratio = row.get("maximum_peer_ratio")
        if not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"{key}: invalid peer target")
    return value


def evaluate(reg, evidence_dir, target_group):
    gate = comparator()
    folder = Path(evidence_dir)
    cache = {}
    rows = []

    def read(suite, name):
        key = (suite, name)
        path = folder / suite / f"{name}.json"
        if not path.exists():
            path = folder / suite / "normalized" / f"{name}.json"
        if key not in cache:
            cache[key] = gate.load(path) if path.exists() else None
        return cache[key]

    for item in reg["cases"]:
        if item["target_group"] != target_group:
            continue
        suite, case_id, peer_engine = item["suite_id"], item["id"], item["peer"]
        result = {"target_group": target_group, "archetype": item.get("archetype"), "suite_id": suite, "id": case_id, "peer": peer_engine,
                  "maximum_peer_ratio": item["maximum_peer_ratio"]}
        try:
            current = read(suite, "affon")
            peer = read(suite, peer_engine)
            baseline = read(suite, "baseline-affon")
            def one(report):
                matches = [case for case in report["cases"] if case["id"] == case_id]
                if not matches:
                    return None
                if len(matches) != 1:
                    raise ValueError(f"{case_id}: duplicate case")
                return {**report, "cases": matches}
            if current is None or one(current) is None:
                result["peer_status"] = result["self_status"] = "unmeasured"
            else:
                result["peer_status"] = "unmeasured"
                result["self_status"] = "unmeasured"
                if peer is not None and one(peer) is not None:
                    if peer.get("engine") != peer_engine:
                        raise ValueError(f"{case_id}: wrong peer engine")
                    peer_result = gate.compare(one(current), one(peer), kind="peer",
                                               maximum_ratio=item["maximum_peer_ratio"], absolute_ms=0)
                    result["peer_ratio"] = peer_result["cases"][0]["ratio"]
                    result["peer_status"] = "pass" if peer_result["gate_pass"] else "gap"
                if baseline is not None and one(baseline) is not None:
                    self_result = gate.compare(one(current), one(baseline), kind="self",
                                               maximum_ratio=1.10, absolute_ms=0)
                    result["self_ratio"] = self_result["cases"][0]["ratio"]
                    result["self_status"] = "pass" if self_result["gate_pass"] else "regression"
            result["status"] = "pass" if result["peer_status"] == result["self_status"] == "pass" else (
                "measured_failure" if result["peer_status"] == "gap" or result["self_status"] == "regression" else "unmeasured")
        except (ValueError, KeyError, TypeError) as error:
            result["status"] = "invalid_evidence"
            result["error"] = str(error)
        rows.append(result)
    if not rows:
        raise ValueError(f"no registered cases for target group {target_group}")
    archetypes = []
    for name in reg.get("required_archetypes", {}).get(target_group, []):
        members = [row for row in rows if row["archetype"] == name]
        status = "missing_case" if not members else (
            "pass" if all(row["status"] == "pass" for row in members) else
            "measured_failure" if any(row["status"] in ("measured_failure", "invalid_evidence") for row in members) else "unmeasured")
        archetypes.append({"name": name, "status": status, "case_count": len(members)})
    return {"schema_version": 1, "suite": "affon_model_coverage_result", "target_group": target_group, "cases": rows,
            "archetypes": archetypes,
            "gate_pass": all(row["status"] == "pass" for row in rows) and all(row["status"] == "pass" for row in archetypes)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/model-coverage.json")
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--target-group", choices=["mac", "cuda-pc"], required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--allow-unmeasured", action="store_true", help="Inventory only; do not fail for missing evidence")
    a = p.parse_args()
    result = evaluate(registry(a.registry), a.evidence_dir, a.target_group)
    encoded = json.dumps(result, indent=2) + "\n"
    print(encoded, end="")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(encoded)
    acceptable = {"pass", "unmeasured"} if a.allow_unmeasured else {"pass"}
    if any(case["status"] not in acceptable for case in result["cases"]) or any(
        item["status"] not in acceptable for item in result["archetypes"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
