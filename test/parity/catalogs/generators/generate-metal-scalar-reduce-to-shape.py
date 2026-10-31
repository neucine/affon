#!/usr/bin/env python3
"""Maintain a scalar-output reduce-to-shape parity case."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    path = ROOT / 'test/benchmarks/capability-probes.json'
    master = json.loads(path.read_text())
    case = {'id': 'metal/f32/reduce_to_shape/scalar', 'op': 'reduce_to_shape',
            'dtype': 'f32', 'shape': [3, 5], 'target_shape': []}
    if case['id'] not in {row['id'] for row in master['cases']}:
        master['cases'].append(case)
    path.write_text(json.dumps(master, indent=2) + '\n')
    output = Path(__file__).resolve().parent.parent / 'metal-scalar-reduce-to-shape.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': [case]}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
