#!/usr/bin/env python3
"""Retain the four previously failing f32 all-elements statistic cells."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    master = json.loads((ROOT / 'test/benchmark/capability-probes.json').read_text())
    known = {case['id']: case for case in master['cases']}
    ids = [f'metal/f32/{op}/{regime}' for op in ('variance_all', 'std_all')
           for regime in ('long_axis', 'strided/r64_c127')]
    cases = [known[cell] for cell in ids]
    output = Path(__file__).resolve().parent.parent / 'metal-variance-std-corrected-4.json'
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes',
                                  'cases': cases}, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
