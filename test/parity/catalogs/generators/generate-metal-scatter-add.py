#!/usr/bin/env python3
"""Maintain the Metal scatter-add axis and offset-view parity catalog."""
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
        for layout in ('dense', 'offset'):
            for axis in (0, 1):
                for rows in (3, 7, 16, 33):
                    for cols in (3, 17, 64, 129):
                        index_shape = [rows // 2 + 1, cols] if axis == 0 else [rows, cols // 2 + 1]
                        regime = 'small' if layout == 'dense' else 'large'
                        label = f'axis{axis}' if layout == 'dense' else f'offset_axis{axis}'
                        cell = f'metal/{dtype}/scatter_add/{regime}/{label}_r{rows}_c{cols}'
                        case = dict(id=cell, op='scatter_add', dtype=dtype, shape=[rows, cols],
                                    index_shape=index_shape, index_axis=axis)
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-scatter-add-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
