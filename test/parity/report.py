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
             "", "## Checks", "",
             f"- Cases run: {len(cells)}",
             f"- Numerical check passed: {probe['correct']}",
             f"- Operator check and timing passed: {operator['passed']}",
             f"- Program check and timing passed: {program['passed']}",
             f"- Numerical failures: {len(cells) - probe['correct']}",
             f"- Operator failures: {operator['incorrect'] + operator['error']}",
             f"- Program failures: {program['incorrect'] + program['error']}",
             "", "## Cases", "",
             "| Case | Family | Shape | Layout | Numerical | Operator | Affon / PyTorch ms | Program | Affon / PyTorch ms |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in cells:
        cell = row["id"].replace("|", "\\|")
        def status(kind):
            value = row[kind]
            artifact = row.get(kind + "_artifact")
            return f"[{value}]({artifact})" if artifact else value
        def timing(kind):
            a, b = row.get(kind + "_affon_ms"), row.get(kind + "_peer_ms")
            return f"{a:.4g} / {b:.4g}" if a is not None and b is not None else "—"
        axes = row.get('axes', {})
        shape = ' × '.join(map(str, axes.get('shape', []))) or '—'
        lines.append(f"| `{cell}` | {axes.get('operation_family', '—')} | {shape} | {axes.get('layout', '—')} | {row['probe_status']} | {status('operator_peer')} | {timing('operator_peer')} | {status('program_peer')} | {timing('program_peer')} |")
    lines.extend(["", "Numerical results are in [capability.json](capability.json).",
                  "Timing changes are diagnostic. The saved files contain the raw checks, including failures.", ""])
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
