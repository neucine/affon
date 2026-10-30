#!/usr/bin/env python3
"""Maintain Metal inner and outer layer norm view parity cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    master_path = ROOT / 'benchmarks/capability-probes.json'
    policy_path = ROOT / 'benchmarks/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for axis in ('inner', 'outer'):
            for layout in ('strided', 'offset'):
                for rows in (3, 16, 33, 128):
                    for cols in (5, 32, 65, 129):
                        regime = 'outer_axis' if axis == 'outer' else 'strided'
                        cell = f'metal/{dtype}/layer_norm/{regime}/{axis}_{layout}_r{rows}_c{cols}'
                        case = {'id': cell, 'op': 'layer_norm', 'dtype': dtype,
                                'shape': [rows, cols]}
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 128
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-layer-norm-views-128.json',
          {'schema_version': 1, 'suite': 'affon_capability_probes', 'cases': cases})
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
