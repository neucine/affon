#!/usr/bin/env python3
"""Maintain the Metal masked-fill layout and mask-broadcast catalog."""
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
        for regime in ('small_dense', 'broadcast', 'strided', 'offset'):
            for rows in (2, 5, 16, 31):
                for cols in (5, 33, 64, 127):
                    cell = f'metal/{dtype}/masked_fill/{regime}/grid_r{rows}_c{cols}'
                    case = dict(id=cell, op='masked_fill', dtype=dtype, shape=[rows, cols])
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-masked-fill-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
