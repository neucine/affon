#!/usr/bin/env python3
"""Build a stable rerunnable audit of declared Metal canonical gaps."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OPS = {'log_softmax_nll', 'cross_entropy', 'scatter_add', 'topk',
       'reduce_to_shape', 'dot', 'quantized_matmul'}


def main():
    spec = importlib.util.spec_from_file_location('coverage_matrix', ROOT / 'tools/coverage-matrix.py')
    matrix = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix)
    policy = json.loads((ROOT / 'benchmarks/coverage-policy.json').read_text())
    diagnostics = json.loads((ROOT / 'benchmarks/workload-coverage.json').read_text())
    outcomes = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
    master_path = ROOT / 'benchmarks/capability-probes.json'
    master = json.loads(master_path.read_text())
    by_id = {case['id']: case for case in master['cases']}
    output = Path(__file__).resolve().parent.parent / 'metal-unknown-canonical-audit.json'
    established = ({case['id'] for case in json.loads(output.read_text())['cases']}
                   if output.exists() else set())
    cases = []
    for row in matrix.build(policy, diagnostics, outcomes)['cells']:
        cell = row['id']
        parts = cell.split('/')
        if (len(parts) != 4 or parts[0] != 'metal' or parts[1] not in ('f32', 'bf16')
                or parts[2] not in OPS or (row['support'] != 'unknown' and cell not in established)
                or parts[3] == 'invalid_index'
                or (parts[2] == 'reduce_to_shape' and parts[1] == 'bf16')):
            continue
        if cell in by_id:
            cases.append(by_id[cell])
            continue
        examples = [case for key, case in by_id.items() if key.startswith(cell + '/')]
        if not examples:
            continue
        case = {**examples[0], 'id': cell}
        master['cases'].append(case)
        by_id[cell] = case
        cases.append(case)
    if len(cases) < 32:
        raise ValueError(f'expected substantial canonical audit, got {len(cases)}')
    master_path.write_text(json.dumps(master, indent=2) + '\n')
    output.write_text(json.dumps({'schema_version': 1, 'suite': 'affon_capability_probes', 'cases': cases}, indent=2) + '\n')
    topk = [case for case in cases if case['op'] == 'topk']
    output.with_name('metal-unknown-canonical-topk-recheck.json').write_text(json.dumps(
        {'schema_version': 1, 'suite': 'affon_capability_probes', 'cases': topk}, indent=2) + '\n')
    print(f'{len(cases)} canonical cells in {output}')


if __name__ == '__main__':
    main()
