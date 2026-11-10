#!/usr/bin/env python3
"""Resume a full parity catalog with immutable build identity and per-cell evidence."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys

import run as batch
from case import contract_hash, normalize

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def source_hash():
    digest = hashlib.sha256()
    paths = [Path(__file__).resolve(), ROOT / 'test/parity/run.py', ROOT / 'test/parity/case.py',
             ROOT / 'tools/import-coverage-evidence.py']
    for folder in ('workers', 'reference', 'fixtures'):
        paths += [p for p in (ROOT / 'test/parity' / folder).rglob('*') if p.suffix in {'.py', '.ts'}]
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def bind_identity(output, identity, resume):
    path = output / 'run.json'
    if path.exists():
        if not resume:
            raise FileExistsError('existing sweep requires --resume')
        if json.loads(path.read_text()) != identity:
            raise ValueError('sweep identity changed: use a fresh output directory for changed inputs, build, reference, target or protocol')
    elif resume:
        raise ValueError('cannot resume without run.json')
    else:
        atomic_json(path, identity)


def filename(cell):
    return cell.replace('/', '__') + '.json'


def load_probe(output, cell):
    path = output / 'numerical' / filename(cell)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if row.get('id') != cell or 'status' not in row:
        raise ValueError(f'invalid checkpoint: {path}')
    return row


def probe_cases(cases, binary, output, run_one, progress):
    rows = {}
    for case in cases:
        cell = case['id']
        row = load_probe(output, cell)
        if row is None:
            try:
                row = run_one(binary, case)
            except Exception as error:
                row = {'id': cell, 'status': 'probe_exception',
                       'stderr': f'{type(error).__name__}: {error}'}
            if row.get('id') != cell or 'status' not in row:
                raise ValueError(f'probe returned wrong cell: {cell}')
            atomic_json(output / 'numerical' / filename(cell), row)
        rows[cell] = row
        progress(cell, rows)
    return rows


def peer_checkpoint(output, kind, cell):
    passed = output / kind / filename(cell)
    failed = output / (kind + '_failures') / filename(cell)
    if passed.exists() and failed.exists():
        raise ValueError(f'ambiguous peer checkpoint: {cell}')
    path = passed if passed.exists() else failed if failed.exists() else None
    if path is None:
        return None
    report = json.loads(path.read_text())
    if report.get('cell_id') != cell or report.get('kind') != kind:
        raise ValueError(f'invalid peer checkpoint: {path}')
    status = 'error' if 'error' in report else 'passed' if report['correctness']['passed'] else 'incorrect'
    if (status == 'passed') != (path == passed):
        raise ValueError(f'peer status does not match checkpoint directory: {path}')
    return status, path, report


def manifest(cases, probes, output, identity):
    rows = []
    for case in cases:
        cell = case['id']
        probe = probes.get(cell)
        row = {'id': cell, 'axes': normalize(case), 'case_sha256': contract_hash(case),
               'probe_status': probe['status'] if probe else 'not_run', 'probe_artifact': 'capability.json'}
        if probe and 'maximum_error' in probe:
            row.update(probe_maximum_error=probe['maximum_error'], probe_tolerance=probe['tolerance'])
        if probe and 'stderr' in probe:
            row['probe_error'] = probe['stderr']
        for kind in ('operator_peer', 'program_peer'):
            saved = peer_checkpoint(output, kind, cell)
            row[kind] = 'probe_not_correct' if probe and probe['status'] != 'correct' else 'not_run'
            if saved:
                status, path, report = saved
                if report.get('affon_binary_sha256', identity['binary_sha256']) != identity['binary_sha256']:
                    raise ValueError(f'peer binary mismatch: {cell}')
                row[kind] = status
                row[kind + '_artifact'] = str(path.relative_to(output))
                if 'correctness' in report:
                    row[kind + '_affon_ms'] = statistics.median(report['affon_samples_ms'])
                    row[kind + '_peer_ms'] = statistics.median(report['peer_samples_ms'])
                    row[kind + '_ratio'] = report['ratio_of_medians']
                    row[kind + '_maximum_error'] = report['correctness']['max_abs_error']
                    row[kind + '_tolerance'] = report['correctness']['tolerance']
        rows.append(row)
    return {'schema_version': 1, 'suite': 'affon_coverage_batch_run',
            **{k: identity[k] for k in ('catalog_sha256', 'binary_sha256', 'target', 'iterations', 'samples', 'rounds')},
            'source_catalog_sha256': identity['catalog_sha256'], 'cells': rows}


def run(args):
    binary = args.binary.resolve()
    catalog_bytes = args.catalog.read_bytes()
    probes_module = batch.load('sweep_probes', 'probe-capabilities.py')
    cases = probes_module.validate_catalog(json.loads(catalog_bytes))
    if not cases or len({c['id'].split('/')[0] for c in cases}) != 1:
        raise ValueError('need a nonempty single-backend catalog')
    if args.iterations < 1 or args.samples < 3 or args.rounds < 2:
        raise ValueError('need positive iterations, at least three samples and two rounds')
    identity = {'schema_version': 1, 'suite': 'affon_parity_sweep',
                'catalog_sha256': hashlib.sha256(catalog_bytes).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'verification_source_sha256': source_hash(), 'target': batch.target_for(cases),
                'iterations': args.iterations, 'samples': args.samples, 'rounds': args.rounds}
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f'choose a fresh output directory: {output}')
    if args.resume and not output.is_dir():
        raise ValueError('cannot resume missing output directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.seek(0); lock.truncate(); lock.write(f'{socket.gethostname()} {os.getpid()}\n'); lock.flush()
        bind_identity(output, identity, args.resume)
        (output / 'catalog.json').write_bytes(catalog_bytes)
        (output / 'numerical').mkdir(exist_ok=True)
        for kind in ('operator_peer', 'program_peer'):
            for folder in (kind, kind + '_failures'):
                (output / folder).mkdir(exist_ok=True)
        state = {'total': len(cases), 'phase': args.phase, 'requested_phase_finished': False}
        def progress(cell, rows):
            state.update(probe_completed=len(rows), probe_correct=sum(r['status'] == 'correct' for r in rows.values()),
                         probe_failed=sum(r['status'] != 'correct' for r in rows.values()), last_cell=cell,
                         updated_at=datetime.now(timezone.utc).isoformat())
            atomic_json(output / 'progress.json', state)
            if len(rows) % 50 == 0 or rows[cell]['status'] != 'correct':
                print(json.dumps(state), flush=True)
        probes = probe_cases(cases, binary, output, probes_module.run_one, progress)
        capability = {'schema_version': 1, 'suite': 'affon_capability_probe_run',
                      'platform': sys.platform, 'zig_version': subprocess.check_output(['zig', 'version'], text=True).strip(),
                      'affon_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'affon_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                      **{k: identity[k] for k in ('binary_sha256', 'catalog_sha256')}, 'cases': list(probes.values())}
        atomic_json(output / 'capability.json', capability)
        atomic_json(output / 'manifest.json', manifest(cases, probes, output, identity))
        if args.phase in ('peers', 'all'):
            benchmark = batch.load('sweep_benchmark', 'benchmark-cells-pytorch.py')
            for kind in ('operator_peer', 'program_peer'):
                state['phase'] = kind
                for index, case in enumerate(cases, 1):
                    cell = case['id']
                    if probes[cell]['status'] != 'correct':
                        continue
                    saved = peer_checkpoint(output, kind, cell)
                    if saved is None:
                        try:
                            report = benchmark.one(binary, case, args.iterations, args.samples, args.rounds, identity['target'], kind)
                        except Exception as error:
                            report = {'cell_id': cell, 'kind': kind, 'error': f'{type(error).__name__}: {error}'}
                        passed = report.get('correctness', {}).get('passed') is True
                        folder = kind if passed else kind + '_failures'
                        atomic_json(output / folder / filename(cell), report)
                    state.update(last_cell=cell, peer_catalog_position=index, updated_at=datetime.now(timezone.utc).isoformat())
                    atomic_json(output / 'progress.json', state)
                    if index % 25 == 0:
                        print(json.dumps(state), flush=True)
            importer = batch.load('sweep_importer', 'import-coverage-evidence.py')
            outcomes = importer.import_probe(output / 'capability.json', set(probes))
            for kind in ('operator_peer', 'program_peer'):
                for case in cases:
                    saved = peer_checkpoint(output, kind, case['id'])
                    if saved and saved[0] == 'passed':
                        importer.import_benchmark(saved[1], outcomes)
            for value in outcomes.values():
                value.pop('probe_contract', None); value.pop('peer_context', None)
            atomic_json(output / 'outcomes.json', outcomes)
        final = manifest(cases, probes, output, identity)
        atomic_json(output / 'manifest.json', final)
        subprocess.run([sys.executable, str(ROOT / 'test/parity/report.py'), str(output / 'manifest.json'),
                        '--output', str(output / 'report.md')], check=True)
        state.update(requested_phase_finished=True,
                     all_checks_passed=all((r['probe_status'], r['operator_peer'], r['program_peer']) ==
                                          ('correct', 'passed', 'passed') for r in final['cells']))
        atomic_json(output / 'progress.json', state)
        print(json.dumps(state), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=('probe', 'peers', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
