#!/usr/bin/env python3
"""Maintain the Metal fused log-softmax NLL parity catalog."""
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
            for rows in (2, 4, 16, 64):
                for classes in (5, 9, 31, 32, 33, 63, 64, 65):
                    regime = 'small' if layout == 'dense' else 'large'
                    label = 'dense' if layout == 'dense' else 'offset'
                    cell = f'metal/{dtype}/log_softmax_nll/{regime}/{label}_r{rows}_c{classes}'
                    case = dict(id=cell, op='log_softmax_nll', dtype=dtype, shape=[rows, classes])
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-log-softmax-nll-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
