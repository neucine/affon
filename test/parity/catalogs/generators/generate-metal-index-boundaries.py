#!/usr/bin/env python3
"""Maintain Metal gather and index-select boundary parity cases."""
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
        for op in ('gather', 'index_select'):
            for layout in ('strided', 'offset'):
                for axis in (0, 1):
                    for rows in (2, 8, 32, 65):
                        for cols in (7, 31, 63, 127):
                            indices = (rows if axis == 0 else cols) // 2 + 1
                            regime = 'strided' if layout == 'strided' else 'large'
                            if op == 'gather':
                                label = f'axis{axis}' if layout == 'strided' else f'offset_axis{axis}'
                                cell = f'metal/{dtype}/{op}/{regime}/{label}_r{rows}_c{cols}'
                                index_shape = [indices, cols] if axis == 0 else [rows, indices]
                            else:
                                cell = f'metal/{dtype}/{op}/{regime}/{layout}_axis{axis}_r{rows}_c{cols}_i{indices}'
                                index_shape = [indices]
                            case = dict(id=cell, op=op, dtype=dtype, shape=[rows, cols],
                                        index_shape=index_shape, index_axis=axis)
                            cases.append(case)
                            if cell not in known:
                                master['cases'].append(case)
                            policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 256
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-index-boundaries-256.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
