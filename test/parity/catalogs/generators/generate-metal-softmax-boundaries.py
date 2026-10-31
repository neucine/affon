#!/usr/bin/env python3
"""Maintain Metal inner- and outer-axis softmax boundary parity cases."""
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
        for op in ('softmax', 'log_softmax'):
            for axis in ('inner', 'outer'):
                for layout in ('strided', 'offset'):
                    regime = 'outer_axis' if axis == 'outer' else ('strided' if layout == 'strided' else 'ragged')
                    for rows in (1, 7, 64, 129):
                        for cols in (17, 63, 127, 257):
                            cell = f'metal/{dtype}/{op}/{regime}/{layout}_r{rows}_c{cols}'
                            case = dict(id=cell, op=op, dtype=dtype, shape=[rows, cols])
                            cases.append(case)
                            if cell not in known:
                                master['cases'].append(case)
                            policy['variants'][cell] = {'dense_inputs': False}
    assert len(cases) == 256
    write(master_path, master)
    write(policy_path, policy)
    write(Path(__file__).resolve().parent.parent / 'metal-softmax-boundaries-256.json',
          dict(schema_version=1, suite='affon_capability_probes', cases=cases))
    print(f'{len(cases)} cells, {len(master["cases"])} master cases')


if __name__ == '__main__':
    main()
