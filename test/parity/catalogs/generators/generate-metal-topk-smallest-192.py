#!/usr/bin/env python3
"""Maintain Metal top-k smallest-value and extreme-k parity cases."""
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
    master['cases'] = [case for case in master['cases'] if not
                       (case['op'] == 'topk' and case['id'].startswith('metal/') and '/large/' in case['id'])]
    policy['variants'] = {cell: options for cell, options in policy['variants'].items() if not
                          (cell.startswith('metal/') and '/topk/large/' in cell)}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('dense', 'strided', 'offset'):
            for axis in (0, 1):
                for rows in (8, 32, 64, 128):
                    for cols in (7, 31, 65, 257):
                        k = 1 if axis == 0 else cols - 1
                        cell = f'metal/{dtype}/topk/large/{layout}_axis{axis}_r{rows}_c{cols}_k{k}_smallest'
                        case = dict(id=cell, op='topk', dtype=dtype, shape=[rows, cols],
                                    axis=axis, k=k, largest=False)
                        cases.append(case)
                        master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-topk-smallest-192.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
