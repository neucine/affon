#!/usr/bin/env python3
"""Maintain the Metal indexed loss shape grid in the policy and probe catalogs."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
WIDTHS = (3, 5, 9, 15, 17, 23, 30, 34, 47, 62, 66, 95, 126, 130, 191, 254)
BATCHES = (2, 4, 6, 10)


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    master_path = ROOT / 'benchmarks/capability-probes.json'
    policy_path = ROOT / 'benchmarks/coverage-policy.json'
    catalog_path = Path(__file__).resolve().parent.parent / 'metal-indexed-loss-grid-128.json'
    master = json.loads(master_path.read_text())
    policy = json.loads(policy_path.read_text())
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for batch in BATCHES:
            for width in WIDTHS:
                regime = 'small' if batch <= 4 else 'large'
                cell = f'metal/{dtype}/cross_entropy_indexed/{regime}/grid_b{batch}_c{width}'
                case = dict(id=cell, op='cross_entropy_indexed', dtype=dtype,
                            shape=[batch, width], index_shape=[batch])
                cases.append(case)
                if cell not in known:
                    master['cases'].append(case)
                policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(catalog_path, dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
