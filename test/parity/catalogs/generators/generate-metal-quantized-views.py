#!/usr/bin/env python3
"""Maintain the Metal quantized matmul transposed and offset view catalog."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    master_path = ROOT / 'test/benchmarks/capability-probes.json'
    policy_path = ROOT / 'test/benchmarks/coverage-policy.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('transposed', 'offset'):
            for rows in (1, 3, 17, 64):
                for inner in (32, 64, 96, 128):
                    for outputs in (3, 19):
                        bits = (2, 4, 8)[(rows + inner // 32 + outputs) % 3]
                        group = (4, 8, 16, 32)[(rows + outputs) % 4]
                        regime = 'transposed' if layout == 'transposed' else 'ragged'
                        label = 'view' if layout == 'transposed' else 'offset_view'
                        cell = f'metal/{dtype}/quantized_matmul/{regime}/{label}_m{rows}_k{inner}_n{outputs}_b{bits}_g{group}'
                        case = dict(id=cell, op='quantized_matmul', dtype=dtype,
                                    shape=[rows, inner, outputs], bits=bits, group_size=group)
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-quantized-views-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
