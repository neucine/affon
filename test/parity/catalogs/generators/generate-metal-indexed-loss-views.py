#!/usr/bin/env python3
"""Maintain the Metal indexed cross entropy layout parity batch."""
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
    master['cases'] = [case for case in master['cases'] if not (
        case['op'] == 'cross_entropy_indexed' and ('/view_b' in case['id'] or '_view_b' in case['id']))]
    policy['variants'] = {cell: options for cell, options in policy['variants'].items()
                          if not (cell.split('/')[2] == 'cross_entropy_indexed' and
                                  ('/view_b' in cell or '_view_b' in cell))}
    known = {case['id'] for case in master['cases']}
    cases = []
    for dtype in ('f32', 'bf16'):
        for layout in ('strided', 'offset'):
            for batch in (2, 5, 16, 33):
                for width in (7, 31, 32, 33, 63, 64, 65, 129):
                    regime = 'strided' if layout == 'strided' else 'large'
                    cell = f'metal/{dtype}/cross_entropy_indexed/{regime}/{layout}_view_b{batch}_c{width}'
                    case = dict(id=cell, op='cross_entropy_indexed', dtype=dtype,
                                shape=[batch, width], index_shape=[batch])
                    cases.append(case)
                    if cell not in known:
                        master['cases'].append(case)
                    policy['variants'][cell] = {'dense_inputs': False}
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-indexed-loss-views-128.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
