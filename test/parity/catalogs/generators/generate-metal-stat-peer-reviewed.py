#!/usr/bin/env python3
"""Regenerate the reviewed BF16 outer-axis variance peer rerun catalog."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
IDS = {
    'metal/bf16/variance_axis/outer_axis/strided_r7_c127',
    'metal/bf16/variance_axis/outer_axis/strided_r7_c257',
    'metal/bf16/variance_axis/outer_axis/offset_r2_c257',
    'metal/bf16/variance_axis/outer_axis/offset_r7_c257',
}


def main():
    source = json.loads((HERE / 'metal-stat-boundaries-256.json').read_text())
    cases = [case for case in source['cases'] if case['id'] in IDS]
    assert len(cases) == len(IDS)
    source['cases'] = cases
    (HERE / 'metal-stat-peer-reviewed-4.json').write_text(json.dumps(source, indent=2) + '\n')
    print(f'{len(cases)} reviewed cells')


if __name__ == '__main__':
    main()
