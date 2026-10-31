#!/usr/bin/env python3
"""Maintain Metal indexed cross entropy boundary view parity cases."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WIDTHS = (8, 16, 24, 30, 34, 48, 62, 66, 96, 126, 130, 192, 254, 255, 256, 257)
BATCHES = (1, 7, 64, 129)


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
        for layout in ('strided', 'offset'):
            for batch in BATCHES:
                for width in WIDTHS:
                    regime = 'strided' if layout == 'strided' else 'large'
                    cell = f'metal/{dtype}/cross_entropy_indexed/{regime}/{layout}_view_b{batch}_c{width}'
                    case = dict(id=cell, op='cross_entropy_indexed', dtype=dtype,
                                shape=[batch, width], index_shape=[batch])
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 256
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-indexed-loss-boundaries-256.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
