#!/usr/bin/env python3
"""Audit a complete parity batch before replacing aggregate evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(batch: Path) -> None:
    manifest = json.loads((batch / 'manifest.json').read_text())
    catalog_bytes = (batch / 'catalog.json').read_bytes()
    catalog = json.loads(catalog_bytes)
    cases = catalog['cases']
    expected = [case['id'] for case in cases]
    assert len(expected) == len(set(expected)), 'duplicate catalog IDs'
    assert hashlib.sha256(catalog_bytes).hexdigest() == manifest['catalog_sha256']
    rows = manifest['cells']
    assert [row['id'] for row in rows] == expected, 'manifest/catalog ID mismatch'
    probe = json.loads((batch / 'capability.json').read_text())
    assert [row['id'] for row in probe['cases']] == expected, 'probe/catalog ID mismatch'
    assert probe['binary_sha256'] == manifest['binary_sha256']
    assert all(row['status'] == 'correct' for row in probe['cases'])
    for kind in ('operator_peer', 'program_peer'):
        found = {path.name for path in (batch / kind).glob('*.json')}
        wanted = {cell.replace('/', '__') + '.json' for cell in expected}
        assert found == wanted, f'{kind}: missing/extra reports: {len(wanted - found)}/{len(found - wanted)}'
        assert not list((batch / f'{kind}_failures').glob('*.json')), f'{kind}: failure reports'
        for row in rows:
            cell = row['id']
            assert row['probe_status'] == 'correct' and row[kind] == 'passed', cell
            report = json.loads((batch / row[f'{kind}_artifact']).read_text())
            assert report['cell_id'] == cell and report['kind'] == kind, cell
            assert report['target'] == manifest['target'], cell
            assert report['affon_binary_sha256'] == manifest['binary_sha256'], cell
            assert report['peer_engine'] == 'pytorch' and report['peer_version'] == '2.10.0', cell
            assert report['correctness']['passed'], cell
            assert report['contract']['iterations_per_sample'] == manifest['iterations'], cell
            assert report['contract']['rounds'] == manifest['rounds'], cell
            rounds = report['rounds']
            assert len(rounds) == manifest['rounds'], cell
            for n, round_result in enumerate(rounds):
                assert round_result['round'] == n and round_result['output_hash_matches'], cell
                assert len(round_result['affon_samples_ms']) == manifest['samples'], cell
                assert len(round_result['peer_samples_ms']) == manifest['samples'], cell
        print(f'{kind}: {len(found)} complete passing reports')
    print(f'target={manifest["target"]} binary={manifest["binary_sha256"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    audit(parser.parse_args().batch)
