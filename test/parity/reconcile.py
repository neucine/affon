#!/usr/bin/env python3
"""Create a separately audited evidence collection with an explicit reviewed rerun."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from audit_complete_batch import audit, require
from case import contract_hash, normalize
import report
import sweep


def reconcile(base, replacement, output, expected_catalog, reason, binary=None, peer_version=None):
    base, replacement, output = base.resolve(), replacement.resolve(), output.resolve()
    require(bool(reason.strip()), 'review reason is required')
    require(not output.exists(), 'choose a fresh reconciliation output')
    require(json.loads((base / 'progress.json').read_text()).get('requested_phase_finished') is True,
            'base sweep is still in progress')
    audit(replacement, peer_version=peer_version, binary=binary)
    original = json.loads((base / 'manifest.json').read_text())
    revised = json.loads((replacement / 'manifest.json').read_text())
    for key in ('binary_sha256', 'target', 'iterations', 'samples', 'rounds'):
        require(original[key] == revised[key], 'replacement changed ' + key)
    base_catalog = json.loads((base / 'catalog.json').read_text())
    cases = {c['id']: c for c in base_catalog['cases']}
    wanted = json.loads(expected_catalog.read_text())['cases']
    require(len(cases) == len(wanted) and set(cases) == {c['id'] for c in wanted}, 'base does not cover expected scope')
    for case in wanted:
        require(normalize(case) == normalize(cases[case['id']]) and
                contract_hash(case) == contract_hash(cases[case['id']]), 'base changed expected contract')
    replacement_cases = json.loads((replacement / 'catalog.json').read_text())['cases']
    replacement_ids = {c['id'] for c in replacement_cases}
    require(replacement_ids.issubset(cases), 'replacement has cells outside base scope')
    base_probe = json.loads((base / 'capability.json').read_text())
    probes = {r['id']: r for r in base_probe['cases']}
    revised_probes = {r['id']: r for r in json.loads((replacement / 'capability.json').read_text())['cases']}
    for case in replacement_cases:
        cell = case['id']
        require(contract_hash(case) == contract_hash(cases[cell]) and normalize(case) == normalize(cases[cell]),
                'replacement changed case contract: ' + cell)
        for field in ('status', 'shape', 'input_shapes', 'output_shape', 'output_dtype', 'outputs',
                      'selected_path', 'compile_options', 'tolerance', 'checksum_tolerance'):
            require(probes[cell].get(field) == revised_probes[cell].get(field),
                    'replacement changed numerical contract: ' + cell + ' ' + field)
    # Sources remain immutable. The result is a derived collection, never a
    # resumed sweep or a claim that all reports share one verifier snapshot.
    shutil.copytree(base, output)
    for name in ('run.json', 'run.lock', 'outcomes.json', 'audit-complete.json', 'progress.json'):
        (output / name).unlink(missing_ok=True)
    provenance = {'suite': 'affon_reviewed_parity_reconciliation', 'reason': reason,
                  'sources': {}, 'replacements': []}
    for label, source in (('base', base), ('replacement', replacement)):
        provenance['sources'][label] = {'path': str(source), 'files': {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('catalog.json', 'manifest.json', 'capability.json')}}
        if (source / 'run.json').exists():
            provenance['sources'][label]['run_identity'] = json.loads((source / 'run.json').read_text())
    for cell in sorted(replacement_ids):
        for kind in ('operator_peer', 'program_peer'):
            filename = sweep.filename(cell)
            old = []
            for folder in (kind, kind + '_failures'):
                path = base / folder / filename
                if path.exists():
                    old.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                (output / folder / filename).unlink(missing_ok=True)
            source = replacement / kind / filename
            (output / kind).mkdir(exist_ok=True)
            shutil.copy2(source, output / kind / filename)
            provenance['replacements'].append({'id': cell, 'kind': kind, 'prior_reports': old,
                                              'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    result = sweep.manifest(base_catalog['cases'], probes, output, original)
    result['reconciliation'] = 'reconciliation.json'
    sweep.atomic_json(output / 'manifest.json', result)
    sweep.atomic_json(output / 'reconciliation.json', provenance)
    summary = audit(output, expected_catalog=expected_catalog, peer_version=peer_version, binary=binary)
    sweep.atomic_json(output / 'audit-complete.json', summary)
    (output / 'report.md').write_text(report.render(result))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'replacement', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--expected-catalog', required=True, type=Path)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--peer-version')
    args = parser.parse_args()
    reconcile(args.base, args.replacement, args.output, args.expected_catalog,
              args.reason, args.binary, args.peer_version)
