#!/usr/bin/env python3
"""Maintain the canonical odd-width vector dot cases."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    path = ROOT / 'test/benchmark/capability-probes.json'
    master = json.loads(path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype, width in (('f32', 257), ('bf16', 259)):
        case = {'id': f'metal/{dtype}/dot/ragged', 'op': 'dot',
                'dtype': dtype, 'shape': [width]}
        cases.append(case)
        if case['id'] not in known:
            master['cases'].append(case)
    path.write_text(json.dumps(master, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-ragged-dot-2.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    print(f'{len(cases)} cells in {output}')


if __name__ == '__main__':
    main()
