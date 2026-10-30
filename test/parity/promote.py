#!/usr/bin/env python3
"""Promote reviewed complete parity cells into the current coverage ledger."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'benchmarks/results'


def importer():
    spec = importlib.util.spec_from_file_location('coverage_importer', ROOT / 'tools/import-coverage-evidence.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def promote(batch, ledger_path, archive=None, complete_only=False, exclude_ids=()):
    batch = batch.resolve()
    manifest = json.loads((batch / 'manifest.json').read_text())
    all_rows = manifest['cells']
    rows = [row for row in all_rows if (row['probe_status'], row['operator_peer'], row['program_peer']) ==
            ('correct', 'passed', 'passed')] if complete_only else all_rows
    if set(exclude_ids) - {row['id'] for row in all_rows}:
        raise ValueError('excluded cell is absent from the batch')
    rows = [row for row in rows if row['id'] not in exclude_ids]
    if not rows or any((row['probe_status'], row['operator_peer'], row['program_peer']) !=
                       ('correct', 'passed', 'passed') for row in rows):
        raise ValueError('batch has incomplete parity evidence; review failures first')
    if len({row['id'] for row in all_rows}) != len(all_rows):
        raise ValueError('batch has duplicate cells')
    existing = json.loads(ledger_path.read_text())
    ids = {row['id'] for row in rows}
    if len(ids) != len(rows):
        raise ValueError('batch has duplicate cells')
    if archive is None and ids.intersection(existing):
        raise ValueError('batch has already linked cells; provide --archive for a reviewed replacement')
    if archive is not None:
        archive = archive.resolve()
        if archive.exists() or not ids.issubset(existing):
            raise ValueError('replacement archive exists or batch contains new cells')
    module = importer()
    fresh = module.import_probe(batch / 'capability.json', {row['id'] for row in all_rows})
    fresh = {cell: value for cell, value in fresh.items() if cell in ids}
    if set(fresh) != ids:
        raise ValueError('numerical probe did not establish every cell')
    for cell, value in fresh.items():
        if cell.endswith('/view_only') and any(node['implementation_id'] == 'metal_contiguous'
                                                for node in value['probe_contract']['path']):
            raise ValueError(f'{cell}: selected materializing path cannot establish view_only')
    copies = []
    for row in rows:
        cell = row['id']
        for kind, aggregate in (('operator_peer', 'cell-peer-current-mac-2026-10-29'),
                                ('program_peer', 'cell-program-peer-mac-2026-10-29')):
            source = batch / row[kind + '_artifact']
            module.import_benchmark(source, fresh)
            destination = RESULTS / aggregate / (cell.replace('/', '__') + '.json')
            if archive is None and destination.exists():
                raise FileExistsError(destination)
            if archive is not None and not destination.exists():
                raise FileNotFoundError(destination)
            copies.append((cell, kind, source, destination))
    if archive is not None:
        archive.mkdir(parents=True)
        for _, kind, _, destination in copies:
            saved = archive / kind / destination.name
            saved.parent.mkdir(exist_ok=True)
            shutil.copy2(destination, saved)
    for cell, kind, source, destination in copies:
        shutil.copy2(source, destination)
        fresh[cell]['evidence'][kind]['artifact'] = str(destination.resolve())
    for cell, value in fresh.items():
        value.pop('probe_contract')
        value.pop('peer_context', None)
        existing[cell] = value
    ledger_path.write_text(json.dumps(existing, indent=2) + '\n')
    print(f'promoted {len(rows)} complete cells from {batch}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--ledger', type=Path,
                        default=RESULTS / 'coverage-outcomes-current-mac-2026-10-29.json')
    parser.add_argument('--archive', type=Path,
                        help='archive existing aggregate reports before replacing reviewed cells')
    parser.add_argument('--complete-only', action='store_true',
                        help='promote only cells with complete probe, operator and Program parity')
    parser.add_argument('--exclude-id', action='append', default=[],
                        help='reviewed diagnostic cell to retain without promotion')
    args = parser.parse_args()
    promote(args.batch, args.ledger, args.archive, args.complete_only, args.exclude_id)


if __name__ == '__main__':
    main()
