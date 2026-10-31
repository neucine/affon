#!/usr/bin/env python3
"""Probe both sides of declared compiler dispatch boundaries."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def probe_module():
    spec = importlib.util.spec_from_file_location("capability_probes", ROOT / "tools/probe-capabilities.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(manifest, binary, run_one):
    if manifest.get("schema_version") != 1 or manifest.get("suite") != "affon_dispatch_boundaries":
        raise ValueError("invalid dispatch boundary manifest")
    seen, rows, cache = set(), [], {}
    for boundary in manifest.get("boundaries", []):
        identifier = boundary["id"]
        if identifier in seen:
            raise ValueError(f"duplicate boundary: {identifier}")
        seen.add(identifier)
        source_ref = boundary["source"].split(":", 1)[0]
        source_root, separator, source_relative = source_ref.partition("/")
        if not separator or source_root not in {"affon", "compute"}:
            raise ValueError(f"{identifier}: invalid source reference")
        source_path = (ROOT if source_root == "affon" else ROOT.parent / "compute") / source_relative
        if not source_path.is_file() or not boundary.get("source_assertion") or boundary["source_assertion"] not in source_path.read_text():
            raise ValueError(f"{identifier}: dispatch source changed; review boundary policy")
        sides = []
        for side in ("below", "at"):
            case = boundary[side]
            key = json.dumps(case, sort_keys=True)
            if key not in cache:
                cache[key] = run_one(binary, case)
            sides.append(cache[key])
        below, at = sides
        below_path = [node["implementation_id"] for node in below.get("selected_path", [])]
        at_path = [node["implementation_id"] for node in at.get("selected_path", [])]
        status = "probe_failed" if below.get("status") != "correct" or at.get("status") != "correct" else (
            "expected_transition" if below_path != at_path and boundary["expected_at_implementation"] in at_path else
            "unobserved_transition" if below_path == at_path else "unexpected_transition")
        rows.append({"id": identifier, "source": boundary["source"], "predicate": boundary["predicate"],
                     "status": status, "below": {"case": boundary["below"], "status": below["status"], "path": below_path},
                     "at": {"case": boundary["at"], "status": at["status"], "path": at_path}})
    return {"schema_version": 1, "suite": "affon_dispatch_boundary_audit",
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "summary": {"boundaries": len(rows), "expected_transition": sum(row["status"] == "expected_transition" for row in rows),
                        "unobserved_transition": sum(row["status"] == "unobserved_transition" for row in rows),
                        "failed": sum(row["status"] in {"probe_failed", "unexpected_transition"} for row in rows)},
            "boundaries": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test/benchmarks/dispatch-boundaries.json")
    parser.add_argument("--binary", type=Path, default=ROOT / "zig-out/bin/affon")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(json.loads(args.manifest.read_text()), args.binary, probe_module().run_one)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
