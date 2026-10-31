#!/usr/bin/env python3
"""Maintain Metal dense soft-target loss parity at row and column boundaries."""
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
        for op in ('cross_entropy', 'log_softmax_nll'):
            for stride in ('rows', 'cols'):
                for rows in (2, 7, 33, 65):
                    for classes in (17, 32, 63, 64, 66, 127, 128, 129):
                        cell = f'metal/{dtype}/{op}/strided/{stride}_r{rows}_c{classes}'
                        case = dict(id=cell, op=op, dtype=dtype, shape=[rows, classes])
                        cases.append(case)
                        if cell not in known:
                            master['cases'].append(case)
                        policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-soft-loss-boundaries-256.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
