#!/usr/bin/env python3
"""Refresh a sweep's derived manifest and report from retained checkpoints."""
import argparse
import json
from pathlib import Path
import tempfile

import report
import sweep


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name+'-', delete=False) as temporary:
        temporary.write(content)
    Path(temporary.name).replace(path)


def refresh(output):
    output = output.resolve()
    cases = json.loads((output / 'catalog.json').read_text())['cases']
    identity = json.loads((output / 'run.json').read_text())
    probes = {case['id']: saved for case in cases
              if (saved := sweep.load_probe(output, case['id'])) is not None}
    result = sweep.manifest(cases, probes, output, identity)
    state = json.loads((output / 'progress.json').read_text())
    result['checkpoint_updated_at'] = state.get('updated_at')
    result['in_progress'] = not state.get('requested_phase_finished', False)
    if state.get('interrupted'):
        result['in_progress'] = False
        result['interrupted'] = state.get('stop_reason', 'Validation was interrupted.')
    atomic_text(output / 'manifest.json', json.dumps(result, indent=2)+'\n')
    atomic_text(output / 'report.md', report.render(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    args = parser.parse_args()
    refresh(args.batch)
