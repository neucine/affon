#!/usr/bin/env python3
"""Compare a parity rerun with a retained run and flag new correctness failures."""
import argparse
import json
from pathlib import Path

TIERS = ('probe_status', 'operator_peer', 'program_peer')
PASS = {'probe_status': 'correct', 'operator_peer': 'passed', 'program_peer': 'passed'}
CASE_IDENTITY = ('device', 'dtype', 'operation', 'shape', 'layout', 'regime', 'variant', 'scope', 'piece')


def compare(previous, current, timing_factor=1.5):
    old = {row['id']: row for row in previous['cells']}
    new = {row['id']: row for row in current['cells']}
    if len(old) != len(previous['cells']) or len(new) != len(current['cells']):
        raise ValueError('duplicate cell in manifest')
    target_matches = previous['target'] == current['target']
    protocol_matches = all(previous[key] == current[key] for key in ('iterations', 'samples', 'rounds'))
    rows = []
    contract_changes = []
    for cell in sorted(old.keys() & new.keys()):
        now = new[cell]
        before = old[cell]
        definition_changed = (before.get('case_sha256') and now.get('case_sha256') and
                              before['case_sha256'] != now['case_sha256'])
        axes_changed = before.get('axes') and now.get('axes') and any(
            before['axes'].get(key) != now['axes'].get(key) for key in CASE_IDENTITY)
        if definition_changed or axes_changed:
            contract_changes.append(cell)
            continue
        regressions = [tier for tier in TIERS if before[tier] == PASS[tier] and now[tier] != PASS[tier]]
        timing = {}
        for tier in ('operator_peer', 'program_peer'):
            prior_ms, current_ms = before.get(tier + '_affon_ms'), now.get(tier + '_affon_ms')
            prior_ratio, current_ratio = before.get(tier + '_ratio'), now.get(tier + '_ratio')
            if target_matches and protocol_matches and before[tier] == now[tier] == 'passed' and prior_ms and current_ms and prior_ratio and current_ratio:
                timing[tier] = {'affon_change': current_ms / prior_ms,
                                'relative_to_peer_change': current_ratio / prior_ratio,
                                'diagnostic_slowdown': current_ratio / prior_ratio >= timing_factor}
        rows.append({'id': cell, 'correctness_regressions': regressions,
                     'previous': {tier: before[tier] for tier in TIERS},
                     'current': {tier: now[tier] for tier in TIERS}, 'timing': timing})
    return {'schema_version': 1, 'suite': 'affon_parity_regression', 'target': current['target'],
            'timing_factor': timing_factor,
            'previous_target': previous['target'], 'target_matches': target_matches,
            'protocol_matches': protocol_matches,
            'added_cells': sorted(new.keys() - old.keys()),
            'removed_cells': sorted(old.keys() - new.keys()),
            'changed_cases': contract_changes,
            'cells': rows, 'correctness_regressions': sum(bool(row['correctness_regressions']) for row in rows),
            'timing_suspects': sum(any(t['diagnostic_slowdown'] for t in row['timing'].values()) for row in rows)}


def render(result):
    lines = ['# Parity regression check', '', f"Target: `{result['target']}`", '',
             f"- Cells compared: {len(result['cells'])}",
             f"- Added cells: {len(result['added_cells'])}",
             f"- Removed cells: {len(result['removed_cells'])}",
             f"- Changed case definitions: {len(result['changed_cases'])}",
             f"- New correctness failures: {result['correctness_regressions']}",
             f"- Diagnostic timing suspects: {result['timing_suspects']}",
             f"- Target matches: {'yes' if result['target_matches'] else 'no'}",
             f"- Measurement protocol matches: {'yes' if result['protocol_matches'] else 'no'}", '',
             'Timing changes are diagnostic and do not fail this check. Timing is compared only when target and protocol match. No timing baseline is accepted by this comparison.', '',
             '| Cell | Correctness regression | Operator relative timing | Program relative timing |',
             '| --- | --- | ---: | ---: |']
    for row in result['cells']:
        def timing(tier):
            value = row['timing'].get(tier)
            return f"{value['relative_to_peer_change']:.2f}×" if value else '—'
        lines.append(f"| `{row['id']}` | {', '.join(row['correctness_regressions']) or 'none'} | {timing('operator_peer')} | {timing('program_peer')} |")
    if result['added_cells']:
        lines += ['', '## Added cells', ''] + [f'- `{cell}`' for cell in result['added_cells']]
    if result['removed_cells']:
        lines += ['', '## Removed cells', ''] + [f'- `{cell}`' for cell in result['removed_cells']]
    if result['changed_cases']:
        lines += ['', '## Changed case definitions', ''] + [f'- `{cell}`' for cell in result['changed_cases']]
    return '\n'.join(lines) + '\n'


def read_manifest(path):
    return json.loads((path / 'manifest.json' if path.is_dir() else path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous', type=Path)
    parser.add_argument('current', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timing-factor', type=float, default=1.5)
    args = parser.parse_args()
    if args.timing_factor <= 1:
        parser.error('--timing-factor must exceed 1')
    result = compare(read_manifest(args.previous), read_manifest(args.current), args.timing_factor)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'regression.json').write_text(json.dumps(result, indent=2) + '\n')
        (args.output_dir / 'regression.md').write_text(render(result))
    print(json.dumps({'cases_compared': len(result['cells']),
                      'changed_cases': len(result['changed_cases']),
                      'correctness_regressions': result['correctness_regressions'],
                      'timing_suspects': result['timing_suspects']}))
    if result['correctness_regressions']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
