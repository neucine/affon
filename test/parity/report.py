#!/usr/bin/env python3
"""Render a parity batch manifest as an auditable Markdown report."""
import argparse
from collections import Counter
import json
from pathlib import Path


def render(manifest):
    cells = manifest["cells"]
    if len({row["id"] for row in cells}) != len(cells):
        raise ValueError("duplicate cell in parity manifest")
    probe = Counter(row["probe_status"] for row in cells)
    operator = Counter(row["operator_peer"] for row in cells)
    program = Counter(row["program_peer"] for row in cells)
    lines = ["# Parity batch report", "",
             f"Target: `{manifest['target']}`  ",
             f"Catalog SHA-256: `{manifest['catalog_sha256']}`  ",
             f"Binary SHA-256: `{manifest['binary_sha256']}`  ",
             f"Timing: {manifest['iterations']} iterations × {manifest['samples']} samples × {manifest['rounds']} rounds",
             "", "## Coverage", "",
             f"- Candidate cells: {len(cells)}",
             f"- Numerical capability passed: {probe['correct']}",
             f"- Operator parity and timing passed: {operator['passed']}",
             f"- Program parity and timing passed: {program['passed']}",
             f"- Probe failures or incorrect: {len(cells) - probe['correct']}",
             f"- Operator incorrect or error: {operator['incorrect'] + operator['error']}",
             f"- Program incorrect or error: {program['incorrect'] + program['error']}",
             "", "## Cells", "",
             "| Cell | Probe | Operator parity | Affon / peer ms | Program parity | Affon / peer ms |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in cells:
        cell = row["id"].replace("|", "\\|")
        def status(kind):
            value = row[kind]
            artifact = row.get(kind + "_artifact")
            return f"[{value}]({artifact})" if artifact else value
        def timing(kind):
            a, b = row.get(kind + "_affon_ms"), row.get(kind + "_peer_ms")
            return f"{a:.4g} / {b:.4g}" if a is not None and b is not None else "—"
        lines.append(f"| `{cell}` | {row['probe_status']} | {status('operator_peer')} | {timing('operator_peer')} | {status('program_peer')} | {timing('program_peer')} |")
    lines.extend(["", "Each probe result is retained in [capability.json](capability.json).",
                  "Timing ratios are diagnostic; only reports with passing numerical parity are eligible for coverage links.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = render(json.loads(args.manifest.read_text()))
    if args.output:
        args.output.write_text(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
