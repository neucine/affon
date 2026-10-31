#!/usr/bin/env python3
"""Maintain the 192-cell Metal top-k values and indices catalog."""
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
    master['cases'] = [case for case in master['cases'] if not
                       (case['op'] == 'topk' and case['id'].startswith('metal/') and '/small/' in case['id'])]
    policy['variants'] = {cell: options for cell, options in policy['variants'].items() if not
                          (cell.startswith('metal/') and '/topk/small/' in cell)}
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('dense', 'strided', 'offset'):
            for axis in (0, 1):
                for rows in (3, 7, 16, 33):
                    for cols in (3, 17, 64, 129):
                        width = (rows, cols)[axis]
                        k = min(width, 1 + width // 2)
                        cell = f'metal/{dtype}/topk/small/{layout}_axis{axis}_r{rows}_c{cols}_k{k}'
                        case = dict(id=cell, op='topk', dtype=dtype, shape=[rows, cols], axis=axis, k=k)
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-topk-192.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
