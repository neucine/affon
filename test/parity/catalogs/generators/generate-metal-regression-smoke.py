#!/usr/bin/env python3
"""Rebuild the compact Metal regression catalog from declared master cases."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'test/parity'))
from case import families, layout_of
IDS = (
    'metal/f32/add/small_dense',
    'metal/bf16/mul/small_dense',
    'metal/f32/matmul/decode_m1',
    'metal/bf16/matmul/decode_m1',
    'metal/f32/sum_axis/short_axis',
    'metal/bf16/softmax/short_row',
    'metal/f32/variance_axis/long_axis',
    'metal/bf16/index_select/small',
    'metal/f32/scatter_add/small/axis0_r3_c3',
    'metal/bf16/topk/small/dense_axis0_r3_c3_k2',
    'metal/f32/cross_entropy_indexed/small',
    'metal/bf16/quantized_matmul/decode_m1/grid_m1_k32_n1_b2_g16',
)


def main():
    master = json.loads((ROOT / 'test/benchmarks/capability-probes.json').read_text())
    cases = {case['id']: case for case in master['cases']}
    missing = set(IDS) - cases.keys()
    if missing:
        raise ValueError(f'master catalog lacks smoke cells: {sorted(missing)}')
    operation_families = families()
    selected = []
    for cell in IDS:
        case = cases[cell].copy()
        case.update(family=operation_families[case['op']], layout=layout_of(case), scope='operation')
        selected.append(case)
    catalog = {'schema_version': 1, 'suite': 'affon_capability_probes', 'cases': selected}
    output = Path(__file__).resolve().parent.parent / 'metal-regression-smoke.json'
    output.write_text(json.dumps(catalog, indent=2) + '\n')
    print(f'{len(IDS)} Metal regression cells: {output}')


if __name__ == '__main__':
    main()
