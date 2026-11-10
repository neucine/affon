#!/usr/bin/env python3
"""Audit every requested cell and its raw parity evidence before promotion."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from case import contract_hash, normalize

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def error_bound(value, tolerance, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f'{label}: nonfinite error')
    require(isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and math.isfinite(tolerance) and tolerance >= 0, f'{label}: invalid tolerance')
    require(0 <= value <= tolerance, f'{label}: error exceeds tolerance')


def samples(values, count, label):
    require(isinstance(values, list) and len(values) == count, f'{label}: missing samples')
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in values), f'{label}: invalid samples')


def audit_seeded_evidence(batch: Path, seed_source: Path | None = None) -> int:
    path = batch / 'seeded-evidence.json'
    if not path.exists():
        require(seed_source is None, 'seed source provided without provenance')
        return 0
    seed = json.loads(path.read_text())
    require(seed.get('suite') == 'affon_reviewed_sweep_seed', 'unexpected seed provenance')
    current = json.loads((batch / 'run.json').read_text())
    require(seed['current_run_identity'] == current, 'seed current identity mismatch')
    declared_source = Path(seed['source'])
    source = seed_source.resolve() if seed_source is not None else declared_source
    require(json.loads((source / 'run.json').read_text()) == seed['source_run_identity'], 'seed source identity mismatch')
    for key in ('catalog_sha256', 'binary_sha256', 'target', 'iterations', 'samples', 'rounds'):
        require(seed['source_run_identity'][key] == current[key], f'seed changed {key}')
    require(hashlib.sha256((source / 'catalog.json').read_bytes()).hexdigest() == current['catalog_sha256'], 'seed source catalog mismatch')
    numerical = seed['numerical_source']
    require(Path(numerical['path']) == declared_source / 'capability.json', 'unexpected seed numerical path')
    require(hashlib.sha256((source / 'capability.json').read_bytes()).hexdigest() == numerical['sha256'], 'seed numerical source hash mismatch')
    original_probe = json.loads((source / 'capability.json').read_text())
    current_probe = json.loads((batch / 'capability.json').read_text())
    require(original_probe['cases'] == current_probe['cases'], 'seed numerical rows changed')
    cases = {case['id'] for case in json.loads((batch / 'catalog.json').read_text())['cases']}
    invalidated = seed['invalidated_operations']
    require(isinstance(invalidated, list) and all(isinstance(op, str) for op in invalidated), 'invalid seed invalidation list')
    seen = set()
    for entry in seed['seeded_peer_reports']:
        cell, kind = entry['id'], entry['kind']
        require(cell in cases and kind in ('operator_peer', 'program_peer'), 'unexpected seeded report identity')
        require((cell, kind) not in seen, 'duplicate seeded report')
        seen.add((cell, kind))
        require(cell.split('/')[2] not in invalidated, f'{cell}: invalidated operation reused')
        relative = Path(kind) / (cell.replace('/', '__') + '.json')
        require(Path(entry['source']) == declared_source / relative, f'{cell}: unexpected seed source path')
        for label, report_path in (('source', source / relative), ('copy', batch / relative)):
            require(hashlib.sha256(report_path.read_bytes()).hexdigest() == entry['sha256'], f'{cell}: seed {label} hash mismatch')
    return len(seen)


def audit(batch: Path, expected_catalog: Path | None = None, peer_version: str | None = None,
          binary: Path | None = None, seed_source: Path | None = None) -> dict:
    batch = batch.resolve()
    manifest = json.loads((batch / 'manifest.json').read_text())
    catalog_bytes = (batch / 'catalog.json').read_bytes()
    cases = json.loads(catalog_bytes)['cases']
    expected = [case['id'] for case in cases]
    require(bool(expected) and len(expected) == len(set(expected)), 'empty or duplicate catalog IDs')
    require(hashlib.sha256(catalog_bytes).hexdigest() == manifest['catalog_sha256'], 'catalog hash mismatch')
    if expected_catalog is not None:
        declared = json.loads(expected_catalog.read_text())['cases']
        wanted = {case['id']: case for case in declared}
        require(len(wanted) == len(declared), 'duplicate expected catalog IDs')
        require(set(expected) == set(wanted), 'batch does not cover expected catalog scope')
        for case in cases:
            require(contract_hash(case) == contract_hash(wanted[case['id']]) and normalize(case) == normalize(wanted[case['id']]), f"expected case contract mismatch: {case['id']}")
    if binary is not None:
        require(hashlib.sha256(binary.read_bytes()).hexdigest() == manifest['binary_sha256'], 'expected binary hash mismatch')
    for key, minimum in (('iterations', 1), ('samples', 3), ('rounds', 2)):
        require(type(manifest.get(key)) is int and manifest[key] >= minimum, f'invalid {key} protocol')
    rows = manifest['cells']
    require([row['id'] for row in rows] == expected, 'manifest/catalog ID mismatch')
    probe_path = batch / 'capability.json'
    probe = json.loads(probe_path.read_text())
    require([row['id'] for row in probe['cases']] == expected, 'probe/catalog ID mismatch')
    require(probe['binary_sha256'] == manifest['binary_sha256'], 'probe binary mismatch')
    require(probe['catalog_sha256'] == manifest['catalog_sha256'], 'probe catalog hash mismatch')
    for case, row, measured in zip(cases, rows, probe['cases']):
        cell = case['id']
        require(measured['status'] == 'correct' and row['probe_status'] == 'correct', f'{cell}: numerical check incomplete')
        require(measured['shape'] == case['shape'], f'{cell}: probe shape differs from catalog')
        if 'axes' in row:
            require(row['axes'] == normalize(case), f'{cell}: manifest axes mismatch')
        if 'case_sha256' in row:
            require(row['case_sha256'] == contract_hash(case), f'{cell}: manifest case hash mismatch')
        require(measured.get('compile_options', {'dense_inputs': False}) == {'dense_inputs': bool(case.get('dense_inputs', False))}, f'{cell}: compile options differ from catalog')
        error_bound(measured['maximum_error'], measured['tolerance'], cell + ' numerical')
        error_bound(measured['checksum_error'], measured['checksum_tolerance'], cell + ' numerical checksum')
        if 'output_hash_matches' in measured:
            require(measured['output_hash_matches'] is True, f'{cell}: numerical output hash mismatch')
    spec = importlib.util.spec_from_file_location('audit_importer', ROOT / 'tools/import-coverage-evidence.py')
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    outcomes = importer.import_probe(probe_path, set(expected))
    require(set(outcomes) == set(expected), 'incomplete imported numerical evidence')
    declared_version = manifest.get('peer_version')
    if declared_version is not None and peer_version is not None:
        require(declared_version == peer_version, 'manifest peer version mismatch')
    bound_version = peer_version or declared_version
    for kind in ('operator_peer', 'program_peer'):
        found = {path.name for path in (batch / kind).glob('*.json')}
        wanted = {cell.replace('/', '__') + '.json' for cell in expected}
        require(found == wanted, f'{kind}: missing/extra reports: {len(wanted - found)}/{len(found - wanted)}')
        require(not list((batch / f'{kind}_failures').glob('*.json')), f'{kind}: failure reports')
        for row in rows:
            cell = row['id']
            require(row[kind] == 'passed', f'{cell}: {kind} incomplete')
            path = batch / kind / (cell.replace('/', '__') + '.json')
            require((batch / row[f'{kind}_artifact']).resolve() == path.resolve(), f'{cell}: unexpected report path')
            report = json.loads(path.read_text())
            require(report['cell_id'] == cell and report['kind'] == kind, f'{cell}: peer identity mismatch')
            require(report['target'] == manifest['target'], f'{cell}: target mismatch')
            require(report['affon_binary_sha256'] == manifest['binary_sha256'], f'{cell}: peer binary mismatch')
            require(report['peer_engine'] == 'pytorch', f'{cell}: unexpected peer engine')
            version = report.get('peer_version')
            require(isinstance(version, str) and bool(version), f'{cell}: missing peer version')
            if bound_version is None:
                bound_version = version
            require(version == bound_version, f'{cell}: peer version mismatch')
            correctness = report['correctness']
            require(correctness['passed'] is True, f'{cell}: peer correctness not established')
            error_bound(correctness['max_abs_error'], correctness['tolerance'], cell + ' peer')
            error_bound(correctness['checksum_error'], correctness['checksum_tolerance'], cell + ' peer checksum')
            require(report['contract']['iterations_per_sample'] == manifest['iterations'], f'{cell}: iterations mismatch')
            require(report['contract']['rounds'] == manifest['rounds'], f'{cell}: round protocol mismatch')
            rounds = report['rounds']
            require(len(rounds) == manifest['rounds'], f'{cell}: round count mismatch')
            flattened = {'affon_samples_ms': [], 'peer_samples_ms': []}
            for n, result in enumerate(rounds):
                require(result['round'] == n and result['output_hash_matches'] is True, f'{cell}: round identity or output hash mismatch')
                require(isinstance(result.get('order'), list) and len(result['order']) == 2 and set(result['order']) == {'affon', 'pytorch'}, f'{cell}: missing matched round order')
                error_bound(result['max_abs_error'], correctness['tolerance'], cell + ' round')
                error_bound(result['checksum_error'], correctness['checksum_tolerance'], cell + ' round checksum')
                require(result['selected_path'] == report['selected_path'], f'{cell}: round implementation changed')
                for key in flattened:
                    samples(result[key], manifest['samples'], cell + ' ' + key)
                    flattened[key].extend(result[key])
            for key, values in flattened.items():
                require(report[key] == values, f'{cell}: flattened samples differ from raw rounds')
            importer.import_benchmark(path, outcomes)
        print(f'{kind}: {len(found)} complete passing reports')
    seeded_reports = audit_seeded_evidence(batch, seed_source)
    summary = {'suite': 'affon_complete_batch_audit', 'cells': len(expected),
               'seeded_reports_provenance_verified': seeded_reports,
               'expected_scope_bound': expected_catalog is not None, 'expected_binary_bound': binary is not None,
               'target': manifest['target'], 'binary_sha256': manifest['binary_sha256'],
               'catalog_sha256': manifest['catalog_sha256'], 'peer_engine': 'pytorch', 'peer_version': bound_version,
               'iterations': manifest['iterations'], 'samples': manifest['samples'], 'rounds': manifest['rounds'],
               'numerical_passed': len(expected), 'operator_passed': len(expected), 'program_passed': len(expected)}
    print(json.dumps(summary))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--expected-catalog', type=Path)
    parser.add_argument('--peer-version')
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--seed-source', type=Path, help='relocated retained source batch for seed provenance verification')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.batch, args.expected_catalog, args.peer_version, args.binary, args.seed_source)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n')
