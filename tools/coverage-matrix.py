#!/usr/bin/env python3
"""Generate the declared Affon performance coverage matrix from Compute's API."""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPUTE = ROOT.parent / "compute"
sys.path.insert(0, str(ROOT / 'test/parity'))
from case import operation_family


def enum_names(source, name):
    match = re.search(r"pub const " + re.escape(name) + r" = enum\s*\{(.*?)\n\};", source, re.S)
    if not match:
        raise ValueError(f"cannot find {name} enum")
    body = re.sub(r"///?[^\n]*", "", match.group(1))
    if name == "DType":
        body = body.split("pub fn", 1)[0]
    return [item.strip() for item in body.split(",") if re.fullmatch(r"[a-z][a-z0-9_]*", item.strip())]


def build(policy, diagnostics, outcomes=None):
    if policy.get("schema_version") != 1 or policy.get("suite") != "affon_coverage_policy":
        raise ValueError("invalid coverage policy")
    source = (COMPUTE / "src/program/types.zig").read_text()
    operations, dtypes = enum_names(source, "OpTag"), enum_names(source, "DType")
    if set(operations) != set(policy["op_regime"]):
        raise ValueError(f"operation policy drift: missing={sorted(set(operations)-set(policy['op_regime']))}, stale={sorted(set(policy['op_regime'])-set(operations))}")
    if set(dtypes) != set(policy["dtypes"]):
        raise ValueError(f"dtype policy drift: source={dtypes}, policy={policy['dtypes']}")
    if set(policy["backends"]) != {"cpu", "metal", "cuda"}:
        raise ValueError("backend policy needs review")
    if any(group not in policy["regimes"] for group in policy["op_regime"].values()):
        raise ValueError("unknown operation regime group")
    if not policy["required_evidence"] or len(set(policy["required_evidence"])) != len(policy["required_evidence"]):
        raise ValueError("invalid evidence requirements")
    related = {}
    for entry in diagnostics.get("cells", []):
        parts = entry["id"].split("/")
        if len(parts) >= 3:
            related.setdefault(tuple(parts[:2] + [entry["family"]]), []).append(entry["id"])
    outcomes = outcomes or {}
    if not isinstance(outcomes, dict):
        raise ValueError("outcomes must be a mapping keyed by cell ID")
    canonical_ids = {f"{backend}/{dtype}/{op}/{regime}"
                                  for backend in policy["backends"] for dtype in dtypes
                                  for op in operations for regime in policy["regimes"][policy["op_regime"][op]]}
    variants = policy.get("variants", {})
    if any(variant.rsplit("/", 1)[0] not in canonical_ids or
           set(options) != {"dense_inputs"} or type(options["dense_inputs"]) is not bool
           for variant, options in variants.items()):
        raise ValueError("invalid coverage variant policy")
    unknown = set(outcomes) - canonical_ids - set(variants)
    if unknown:
        raise ValueError(f"outcomes contain unknown cells: {sorted(unknown)[:5]}")
    rows = []
    for backend in policy["backends"]:
        for dtype in dtypes:
            for op in operations:
                group = policy["op_regime"][op]
                for regime in policy["regimes"][group]:
                    base_id = f"{backend}/{dtype}/{op}/{regime}"
                    for case_id in [base_id] + [key for key in variants if key.rsplit("/", 1)[0] == base_id]:
                        rows.append(make_row(case_id, group, outcomes, policy, related, backend, dtype, op, variants.get(case_id)))
    counts = Counter(row["support"] for row in rows)
    evidence_counts = {kind: Counter("missing" if row["evidence"][kind] == "missing" else "linked" for row in rows)
                       for kind in policy["required_evidence"]}
    distribution = {}
    for dimension, selector in (("backend", lambda row: row["id"].split("/")[0]),
                                ("dtype", lambda row: row["id"].split("/")[1]),
                                ("family", lambda row: row["family"]),
                                ("backend × dtype", lambda row: " / ".join(row["id"].split("/")[:2]))):
        groups = sorted({selector(row) for row in rows})
        distribution[dimension] = {group: {
            "candidate": sum(selector(row) == group for row in rows),
            "supported": sum(selector(row) == group and row["support"] == "supported" for row in rows),
            "correctness_linked": sum(selector(row) == group and row["evidence"]["correctness"] != "missing" for row in rows),
            "operator_peer_linked": sum(selector(row) == group and row["evidence"]["operator_peer"] != "missing" for row in rows),
        } for group in groups}
    return {"schema_version": 1, "suite": "affon_coverage_matrix", "source": policy["source"],
            "summary": {"operations": len(operations), "dtypes": len(dtypes), "backends": len(policy["backends"]),
                        "cells": len(rows), "support": dict(counts),
                        "distribution": distribution,
                        "evidence": {kind: dict(values) for kind, values in evidence_counts.items()},
                        "related_diagnostic_cells": sum(bool(row["related_diagnostics"]) for row in rows)},
            "cells": rows}


def make_row(case_id, group, outcomes, policy, related, backend, dtype, op, variant):
    outcome = outcomes.get(case_id, {})
    support = outcome.get("support", "unknown")
    if support not in {"unknown", "supported", "unsupported", "not_applicable"}:
        raise ValueError(f"{case_id}: invalid support state")
    if support in {"unsupported", "not_applicable"} and not outcome.get("reason"):
        raise ValueError(f"{case_id}: {support} needs reason")
    evidence = outcome.get("evidence", {})
    if set(evidence) - set(policy["required_evidence"]):
        raise ValueError(f"{case_id}: unknown evidence type")
    if any(not isinstance(value, dict) or not value.get("artifact") or
           not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) or
           not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value.get("contract", "")))
           for value in evidence.values()):
        raise ValueError(f"{case_id}: evidence needs artifact SHA-256 and contract SHA-256")
    if support == "supported" and "correctness" not in evidence:
        raise ValueError(f"{case_id}: supported needs correctness evidence")
    if support in {"unsupported", "not_applicable"} and evidence:
        raise ValueError(f"{case_id}: excluded cell cannot carry performance evidence")
    return {"id": case_id, "device": backend, "dtype": dtype,
                 "operation": op, "operation_family": operation_family(op, group),
                 "family": operation_family(op, group), "regime": case_id.split('/')[3],
                 "variant_name": '/'.join(case_id.split('/')[4:]) or None,
                 "scope": "operation", "variant": variant, "support": support,
                 "reason": outcome.get("reason"),
                 "evidence": {kind: evidence.get(kind, "missing") for kind in policy["required_evidence"]},
                 "related_diagnostics": related.get((backend, dtype, op), []) + related.get((backend, dtype, group), [])}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=ROOT / "test/benchmark/coverage-policy.json")
    parser.add_argument("--diagnostics", type=Path, default=ROOT / "test/benchmark/workload-coverage.json")
    parser.add_argument("--outcomes", type=Path, help="Reviewed per-cell capability and evidence decisions")
    parser.add_argument("--backend", choices=["cpu", "metal", "cuda"])
    parser.add_argument("--dtype", choices=["bf16", "f32", "f64", "u32", "i64"])
    parser.add_argument("--op", help="Canonical operation name")
    parser.add_argument("--details", action="store_true", help="Print all generated cells")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    outcomes = json.loads(args.outcomes.read_text()) if args.outcomes else None
    report = build(json.loads(args.policy.read_text()), json.loads(args.diagnostics.read_text()), outcomes)
    if args.backend or args.dtype or args.op:
        report["cells"] = [row for row in report["cells"] if
                           (not args.backend or row["id"].split("/")[0] == args.backend) and
                           (not args.dtype or row["id"].split("/")[1] == args.dtype) and
                           (not args.op or row["id"].split("/")[2] == args.op)]
        if args.op and not report["cells"]:
            raise ValueError(f"unknown or empty operation: {args.op}")
        report["summary"]["selected_cells"] = len(report["cells"])
        report["summary"]["selected_support"] = dict(Counter(row["support"] for row in report["cells"]))
        report["summary"]["selected_related_diagnostics"] = sum(bool(row["related_diagnostics"]) for row in report["cells"])
    if not args.details:
        report.pop("cells")
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
