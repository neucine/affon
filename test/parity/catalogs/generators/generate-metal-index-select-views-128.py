#!/usr/bin/env python3
"""Maintain Metal index-select axis and source-view parity cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    master_path = ROOT / 'test/benchmark/capability-probes.json'
    policy_path = ROOT / 'test/benchmark/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for axis in (0, 1):
            for layout in ('strided', 'offset'):
                for rows in (3, 7, 16, 33):
                    for cols in (5, 32, 65, 129):
                        indices = [shape // 2 + 1 for shape in (rows, cols)][axis]
                        regime = 'strided' if layout == 'strided' else 'large'
                        cell = f'metal/{dtype}/index_select/{regime}/{layout}_axis{axis}_r{rows}_c{cols}_i{indices}'
                        case = dict(id=cell, op='index_select', dtype=dtype, shape=[rows, cols],
                                    index_shape=[indices], index_axis=axis)
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-index-select-views-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
