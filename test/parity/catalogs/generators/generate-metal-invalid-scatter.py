#!/usr/bin/env python3
"""Maintain invalid-index Metal scatter parity and failure evidence."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    path = ROOT / 'test/benchmarks/capability-probes.json'
    master = json.loads(path.read_text())
    known = {row['id'] for row in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        case = {'id': f'metal/{dtype}/scatter_add/invalid_index',
                'op': 'scatter_add', 'dtype': dtype, 'shape': [3, 3],
                'index_shape': [2, 3], 'index_axis': 0}
        cases.append(case)
        if case['id'] not in known:
            master['cases'].append(case)
    path.write_text(json.dumps(master, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-invalid-scatter-2.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
