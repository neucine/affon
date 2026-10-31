#!/usr/bin/env python3
"""Inventory workload evidence without treating a diagnostic as a performance pass."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOTS = {"affon": ROOT, "compute": ROOT.parent / "compute"}
TIERS = {"operator", "program", "model"}


def inventory(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("suite") != "affon_workload_coverage":
        raise ValueError("invalid workload coverage manifest")
    rows, seen = [], set()
    for cell in manifest.get("cells", []):
        name = cell["id"]
        if name in seen:
            raise ValueError(f"duplicate cell: {name}")
        seen.add(name)
        required = cell["required"]
        if not required or len(set(required)) != len(required) or set(required) - TIERS:
            raise ValueError(f"{name}: invalid required tiers")
        if set(cell["evidence"]) - set(required):
            raise ValueError(f"{name}: evidence outside required tiers")
        tiers = {}
        for tier in required:
            reference = cell["evidence"].get(tier)
            if reference is None:
                tiers[tier] = "missing"
                continue
            root_name, separator, location = reference.partition("/")
            if not separator or root_name not in ROOTS:
                raise ValueError(f"{name}: invalid evidence root: {reference}")
            relative, marker, case_id = location.partition(":")
            path = ROOTS[root_name] / relative
            if not path.is_file():
                tiers[tier] = "unavailable"
            elif case_id and case_id not in path.read_text(errors="replace"):
                tiers[tier] = "unverified_case"
            else:
                tiers[tier] = "diagnostic"
        rows.append({"id": name, "family": cell["family"], "regime": cell["regime"],
                     "tiers": tiers, "missing": [tier for tier, state in tiers.items() if state != "diagnostic"]})
    return {"schema_version": 1, "suite": "affon_workload_inventory", "cells": rows,
            "summary": {"cells": len(rows), "complete_diagnostic_links": sum(not row["missing"] for row in rows),
                        "missing_by_tier": {tier: sum(tier in row["missing"] for row in rows) for tier in sorted(TIERS)}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "test/benchmark/workload-coverage.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inventory(json.loads(args.manifest.read_text()))
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
