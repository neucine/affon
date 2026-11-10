#!/usr/bin/env python3
"""Mirror supported Metal contracts onto CUDA without claiming CUDA support.

The target includes every supported Metal cell in the reviewed ledger. Only the
backend in the ID changes; shapes, layouts, options, and tolerances stay intact.
"""
import argparse
import copy
import sys
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'test/parity'))
from case import contract_hash


def mirror(catalog, ledger):
    cases = {case['id']: case for case in catalog['cases']}
    supported = sorted(cell for cell, row in ledger.items()
                       if cell.startswith('metal/') and row['support'] == 'supported')
    missing = set(supported) - cases.keys()
    if missing:
        raise ValueError(f'supported Metal contracts missing from catalog: {sorted(missing)}')
    return [{**cases[cell], 'id': 'cuda/' + cell.split('/', 1)[1]} for cell in supported]


def smoke(cases):
    # One case per dtype/operation, then representative non-dense cases. Full
    # coverage always comes from the complete catalog, never this smoke subset.
    selected = {}
    groups = set()
    for case in cases:
        parts = case['id'].split('/')
        key = tuple(parts[1:3])
        if key not in groups:
            groups.add(key)
            selected[case['id']] = case
    for layout in ('strided', 'offset', 'transposed', 'batched', 'outer_axis'):
        for dtype in ('f32', 'bf16'):
            found = next((case for case in cases if case['dtype'] == dtype and
                          layout in case['id'] and case['id'] not in selected), None)
            if found:
                selected[found['id']] = found
    return sorted(selected.values(), key=lambda case: case['id'])



def update_plan(catalog, policy, mirrored):
    catalog, policy = copy.deepcopy(catalog), copy.deepcopy(policy)
    existing = {case['id']: case for case in catalog['cases']}
    if len(existing) != len(catalog['cases']):
        raise ValueError('duplicate master case IDs')
    variants = policy.setdefault('variants', {})
    for case in mirrored:
        cell = case['id']
        if cell in existing:
            if contract_hash(existing[cell]) != contract_hash(case):
                raise ValueError(f'CUDA contract conflict: {cell}')
        else:
            catalog['cases'].append(case)
            existing[cell] = case
        if len(cell.split('/')) > 4:
            source = 'metal/' + cell.split('/', 1)[1]
            if source not in variants:
                raise ValueError(f'Metal variant absent from policy: {source}')
            if cell in variants and variants[cell] != variants[source]:
                raise ValueError(f'CUDA variant conflict: {cell}')
            variants[cell] = copy.deepcopy(variants[source])
    return catalog, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=ROOT / 'test/benchmark/capability-probes.json')
    parser.add_argument('--ledger', type=Path, default=ROOT / 'test/benchmark/results/coverage-outcomes-current-mac-2026-10-29.json')
    parser.add_argument('--policy', type=Path, default=ROOT / 'test/benchmark/coverage-policy.json')
    parser.add_argument('--update-plan', action='store_true', help='register mirrored cases and policy variants without changing support evidence')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'test/parity/catalogs')
    args = parser.parse_args()
    catalog_bytes, ledger_bytes = args.catalog.read_bytes(), args.ledger.read_bytes()
    cases = mirror(json.loads(catalog_bytes), json.loads(ledger_bytes))
    if args.update_plan:
        catalog, policy = update_plan(json.loads(catalog_bytes), json.loads(args.policy.read_text()), cases)
        catalog_bytes = (json.dumps(catalog, indent=2) + '\n').encode()
        args.catalog.write_bytes(catalog_bytes)
        args.policy.write_text(json.dumps(policy, indent=2) + '\n')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    affected = [case for case in cases if case['op'] in {
        'sin', 'cos', 'reduce_to_shape', 'quantized_matmul', 'layer_norm', 'rms_norm'}]
    losses = [case for case in cases if case['op'] in {'cross_entropy', 'log_softmax_nll'}]
    for name, selected in (('cuda-metal-parity', cases), ('cuda-metal-smoke', smoke(cases)),
                           ('cuda-metal-gap-regression', affected),
                           ('cuda-metal-loss-regression', losses),
                           ('cuda-metal-loss-smoke', [case for case in smoke(cases)
                                                     if case['op'] in {'cross_entropy', 'log_softmax_nll'}])):
        output = {'schema_version': 1, 'suite': 'affon_capability_probes',
                  'generator': Path(__file__).name,
                  'source_catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
                  'source_ledger_sha256': hashlib.sha256(ledger_bytes).hexdigest(),
                  'cases': selected}
        path = args.output_dir / (name + '.json')
        path.write_text(json.dumps(output, indent=2) + '\n')
        print(f'{path.name}: {len(selected)} cells')


if __name__ == '__main__':
    main()
