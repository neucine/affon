#!/usr/bin/env python3
"""Build a self-contained Weiv HTML view of coverage and peer diagnostics."""
import argparse
from datetime import datetime
import importlib.util
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
WEIV = ROOT.parent / "weiv" / "node_modules/.bin/weiv"


def active_sweeps(results, completed=False):
    batches = []
    for path in sorted(results.glob('parity-*/progress.json')):
        state = json.loads(path.read_text())
        if bool(state.get('requested_phase_finished')) != completed or state.get('interrupted') or not (path.parent / 'run.json').is_file():
            continue
        subprocess.run([sys.executable, str(ROOT / 'test/parity/refresh.py'), str(path.parent)], check=True)
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        rows = manifest['cells']
        batches.append({'name': path.parent.name, 'directory': str(path.parent.resolve()),
                        'target': manifest['target'], 'total': len(rows),
                        'numerical': sum(r['probe_status'] == 'correct' for r in rows),
                        'operator': sum(r['operator_peer'] == 'passed' for r in rows),
                        'program': sum(r['program_peer'] == 'passed' for r in rows),
                        'failed': sum(r['probe_status'] not in ('correct', 'not_run') or
                                      r['operator_peer'] in ('incorrect', 'error') or
                                      r['program_peer'] in ('incorrect', 'error') for r in rows),
                        'updated': manifest.get('checkpoint_updated_at') or 'Not recorded',
                        'checks': {kind: [r['id'] for r in rows if r[field] == passed]
                                   for kind, field, passed in (('correctness', 'probe_status', 'correct'),
                                                              ('operator_peer', 'operator_peer', 'passed'),
                                                              ('program_peer', 'program_peer', 'passed'))},
                        'href': path.parent.name + '/report.md'})
    return batches


def recorded_coverage(matrix, batches):
    """Union reviewed records with completed checkpoints, counting each cell once."""
    rows = {row['id']: row for row in matrix.get('cells', [])}
    kinds = ('correctness', 'operator_peer', 'program_peer')
    recorded = {kind: {cell for cell, row in rows.items() if row['evidence'][kind] != 'missing'}
                for kind in kinds}
    for batch in batches or []:
        for kind, cells in batch.get('checks', {}).items():
            if kind not in recorded or set(cells) - rows.keys():
                raise ValueError('active sweep contains unknown coverage cells or check kinds')
            recorded[kind].update(cells)
    if not rows:
        return {kind: matrix['summary']['evidence'][kind].get('linked', 0) for kind in kinds}, matrix['summary'].get('distribution', {})
    distribution = {}
    for dimension, selector in (('backend', lambda row: row['id'].split('/')[0]),
                                ('dtype', lambda row: row['id'].split('/')[1]),
                                ('family', lambda row: row['family']),
                                ('backend × dtype', lambda row: ' / '.join(row['id'].split('/')[:2]))):
        groups = {name: {**counts, 'correctness_recorded': 0, 'operator_peer_recorded': 0}
                  for name, counts in matrix['summary']['distribution'][dimension].items()}
        for cell, row in rows.items():
            group = groups[selector(row)]
            for kind in ('correctness', 'operator_peer'):
                group[kind + '_recorded'] += cell in recorded[kind]
        distribution[dimension] = groups
    return {kind: len(cells) for kind, cells in recorded.items()}, distribution


def validate_evidence_artifacts(outcomes):
    """Refuse a report whose coverage counts refer to missing or changed files."""
    checked = {}
    for cell, outcome in outcomes.items():
        for kind, link in outcome.get("evidence", {}).items():
            path = Path(link["artifact"])
            if not path.is_file():
                raise ValueError(f"{cell} {kind}: evidence artifact missing: {path}")
            digest = checked.get(path)
            if digest is None:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                checked[path] = digest
            if digest != link["sha256"]:
                raise ValueError(f"{cell} {kind}: evidence artifact changed: {path}")


def validate_measured_links(matrix, cases, raw_paths, kind):
    if "cells" not in matrix:
        return
    measured = {row["id"] for row in cases}
    linked = {row["id"] for row in matrix["cells"] if row["evidence"][kind] != "missing"}
    if measured != linked:
        raise ValueError(f"{kind} measured rows and linked evidence differ: "
                         f"unlinked={sorted(measured-linked)}, unmeasured={sorted(linked-measured)}")
    for row in matrix["cells"]:
        if row["id"] in measured and Path(row["evidence"][kind]["artifact"]).resolve() != raw_paths[row["id"]]:
            raise ValueError(f"{row['id']}: {kind} report uses a different raw artifact")


def validate_summary(summary, raw_directory, kind):
    spec = importlib.util.spec_from_file_location("cell_summary", ROOT / "tools/summarize-cell-performance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = [json.loads(path.read_text()) for path in sorted(raw_directory.glob("*.json"))]
    expected = module.summarize(raw, summary["ratio_limit"], kind)
    normalized = {**summary, "kind": summary.get("kind", "operator_peer")}
    if normalized != expected:
        raise ValueError(f"{kind} summary differs from raw benchmark rounds")


def measured_rows(summary, raw_directory, kind):
    if summary.get("suite") != "affon_cell_diagnostic_summary" or summary.get("kind", "operator_peer") != kind:
        raise ValueError(f"unexpected {kind} summary")
    raw = {}
    raw_paths = {}
    for path in raw_directory.glob("*.json"):
        report = json.loads(path.read_text())
        if report.get("suite") != "affon_cell_benchmark" or report.get("kind") != kind:
            raise ValueError(f"unexpected raw result: {path}")
        if report["cell_id"] in raw:
            raise ValueError(f"duplicate {kind} raw result: {report['cell_id']}")
        raw[report["cell_id"]] = report
        raw_paths[report["cell_id"]] = path.resolve()
    cases = []
    for row in summary["cases"]:
        source = raw.get(row["id"])
        if source is None:
            raise ValueError(f"missing raw result for {row['id']}")
        cases.append(measurement_row(row, source, 'Reviewed record'))
    return cases, raw, raw_paths


def measurement_row(row, source, evidence_state):
    phases = source.get('phase_medians_ms', {})
    return {**row, 'path': ', '.join(node['implementation_id'] for node in source['selected_path']),
            'shape': ' × '.join(map(str, source['contract']['shape'])),
            'graph': ' → '.join(source['contract'].get('program', {}).get('ops', [])),
            'target': source['target'], 'evidence_state': evidence_state,
            'peer_note': ('Peer timing: '+source['contract']['peer_algorithm'].replace('_', ' ')
                          if source['contract'].get('peer_algorithm') else ''),
            'affon_execution_ms': phases.get('affon_execution_ms'),
            'affon_readback_ms': phases.get('affon_readback_ms')}


def checkpoint_measurements(batches, known_ids, ratio_limit=1.25):
    spec = importlib.util.spec_from_file_location('report_importer', ROOT / 'tools/import-coverage-evidence.py')
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    spec = importlib.util.spec_from_file_location('report_summary', ROOT / 'tools/summarize-cell-performance.py')
    summarizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summarizer)
    spec = importlib.util.spec_from_file_location('report_auditor', ROOT / 'test/parity/audit_complete_batch.py')
    auditor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auditor)
    reports = {kind: {} for kind in ('operator_peer', 'program_peer')}
    for batch in sorted(batches or [], key=lambda batch: (batch['updated'], batch['name'])):
        directory = Path(batch['directory'])
        outcomes = importer.import_probe(directory / 'capability.json', known_ids)
        identity = json.loads((directory / 'run.json').read_text())
        for kind in reports:
            for cell in batch['checks'].get(kind, []):
                path = directory / kind / (cell.replace('/', '__') + '.json')
                source = json.loads(path.read_text())
                if source.get('cell_id') != cell or source.get('kind') != kind or source.get('target') != identity['target']:
                    raise ValueError(f'{path}: checkpoint identity mismatch')
                auditor.require(source.get('affon_binary_sha256') == identity['binary_sha256'], f'{cell}: checkpoint binary mismatch')
                auditor.require(outcomes[cell]['probe_contract']['catalog_sha256'] == identity['catalog_sha256'], f'{cell}: checkpoint catalog mismatch')
                correctness = source['correctness']
                for field, tolerance in (('max_abs_error', 'tolerance'), ('checksum_error', 'checksum_tolerance')):
                    auditor.error_bound(correctness[field], correctness[tolerance], cell)
                auditor.require(source['contract']['iterations_per_sample'] == identity['iterations'] and
                                source['contract']['rounds'] == identity['rounds'] and len(source['rounds']) == identity['rounds'],
                                f'{cell}: checkpoint protocol mismatch')
                flattened = {'affon_samples_ms': [], 'peer_samples_ms': []}
                for n, measured in enumerate(source['rounds']):
                    auditor.require(measured['round'] == n and measured['output_hash_matches'] is True and
                                    measured['selected_path'] == source['selected_path'] and
                                    len(measured['order']) == 2 and set(measured['order']) == {'affon', 'pytorch'},
                                    f'{cell}: checkpoint round identity or output mismatch')
                    for field, tolerance in (('max_abs_error', 'tolerance'), ('checksum_error', 'checksum_tolerance')):
                        auditor.error_bound(measured[field], correctness[tolerance], cell)
                    for key in flattened:
                        auditor.samples(measured[key], identity['samples'], cell)
                        flattened[key].extend(measured[key])
                for key, values in flattened.items():
                    auditor.require(source[key] == values, f'{cell}: checkpoint samples differ from raw rounds')
                importer.import_benchmark(path, outcomes)
                reports[kind][cell] = source
    rows = {}
    for kind, raw in reports.items():
        summary = summarizer.summarize(list(raw.values()), ratio_limit, kind)
        rows[kind] = [measurement_row(row, raw[row['id']], 'Completed checkpoint') for row in summary['cases']]
    return rows, reports


def merge_measurements(reviewed, checkpoints):
    # A completed or still-running sweep can predate a reviewed promotion.
    # Use it only for cells that have no current reviewed measurement.
    merged = {row['id']: row for row in checkpoints}
    merged.update({row['id']: row for row in reviewed})
    return sorted(merged.values(), key=lambda row: (row['status'] != 'candidate_gap',
                  -max(0, row['delta_ms']), -row['ratio']))


def data(matrix, summary, raw_directory, boundary_report=None, program_summary=None, program_raw_directory=None, calibration_directory=None, gpu_profiles=None, baseline_evaluations=None, gpu_investigations=None, paired_gpu=None, active_batches=None, coverage_batches=None):
    totals = matrix["summary"]
    if matrix.get("suite") != "affon_coverage_matrix":
        raise ValueError("unexpected matrix suite")
    recorded, current_distribution = recorded_coverage(matrix, active_batches if coverage_batches is None else coverage_batches)
    cases, raw, raw_paths = measured_rows(summary, raw_directory, "operator_peer")
    program_cases = []
    program_raw = {}
    if program_summary is not None:
        if program_raw_directory is None:
            raise ValueError("Program raw directory missing")
        program_cases, program_raw, program_raw_paths = measured_rows(program_summary, program_raw_directory, "program_peer")
    validate_measured_links(matrix, cases, raw_paths, "operator_peer")
    if program_summary is not None:
        validate_measured_links(matrix, program_cases, program_raw_paths, "program_peer")
    checkpoint_batches = active_batches if coverage_batches is None else coverage_batches
    if checkpoint_batches:
        checkpoint_rows, checkpoint_raw = checkpoint_measurements(
            checkpoint_batches, {row['id'] for row in matrix['cells']}, summary['ratio_limit'])
        for kind, reviewed_raw in (('operator_peer', raw), ('program_peer', program_raw)):
            for row in checkpoint_rows[kind]:
                cell = row['id']
                if cell in reviewed_raw and checkpoint_raw[kind][cell] == reviewed_raw[cell]:
                    row['evidence_state'] = 'Reviewed record'
        cases = merge_measurements(cases, checkpoint_rows['operator_peer'])
        program_cases = merge_measurements(program_cases, checkpoint_rows['program_peer'])
        raw.update(checkpoint_raw['operator_peer'])
    if 'cells' in matrix and (len(cases) != recorded['operator_peer'] or len(program_cases) != recorded['program_peer']):
        raise ValueError('timing lists differ from recorded coverage counts')
    baseline_calibrations = []
    calibration_directories = (calibration_directory if isinstance(calibration_directory, list) else
                               [calibration_directory] if calibration_directory is not None else [])
    for folder in calibration_directories:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.json")):
            report = json.loads(path.read_text())
            if report.get("suite") != "affon_cell_baseline_calibration":
                raise ValueError(f"unexpected baseline calibration: {path}")
            baseline_calibrations.append({"id": report["cell_id"],
                "status": "Within-session pass" if report["calibration"]["stable"] else "Too variable",
                "runs": len(report["runs"]),
                "protocol": f"{report['contract'].get('warmup_calls', 20)} warmup · {report['contract']['iterations_per_sample']} calls/sample",
                "run_medians_ms": [row["median_ms"] for row in report["runs"]],
                "spread_ratio": report["calibration"]["observed_run_median_ratio"],
                "spread_ms": report["calibration"]["observed_run_median_span_ms"]})
    baseline_verdicts = []
    if baseline_evaluations is not None and baseline_evaluations.is_dir():
        for path in sorted(baseline_evaluations.glob("*.json")):
            verdict = json.loads(path.read_text())
            if verdict.get("suite") != "affon_cell_baseline_evaluation":
                raise ValueError(f"unexpected baseline evaluation: {path}")
            baseline_verdicts.append({"id": verdict["cell_id"], "accepted": verdict["accepted"],
                "session_medians_ms": verdict["session_medians_ms"],
                "ratio": verdict["between_session"]["ratio"], "reasons": "; ".join(verdict["reasons"])})
    gpu_comparisons = []
    if gpu_profiles:
        if len(gpu_profiles) < 2:
            raise ValueError("GPU variability comparison needs at least two sessions")
        first = gpu_profiles[0]
        for profile in gpu_profiles:
            if profile.get("suite") != "affon_cell_gpu_variability":
                raise ValueError("unexpected GPU variability report")
            for key in ("cell_id", "binary_sha256", "interposer_sha256", "contract"):
                if profile[key] != first[key]:
                    raise ValueError(f"GPU sessions differ in {key}")
            if {json.dumps(row["selected_path"], sort_keys=True) for row in profile["runs"]} != {
                    json.dumps(row["selected_path"], sort_keys=True) for row in first["runs"]}:
                raise ValueError("GPU sessions selected different paths")
        def medians(key):
            return [statistics.median(row[key] for row in profile["runs"]) for profile in gpu_profiles]
        gpu = medians("gpu_command_median_ms")
        host = medians("host_total_median_ms")
        gpu_comparisons.append({"id": first["cell_id"], "sessions": len(gpu_profiles),
            "gpu_medians_ms": gpu, "host_medians_ms": host,
            "gpu_ratio": max(gpu) / min(gpu), "host_ratio": max(host) / min(host)})
    investigation_rows = []
    for index, profile in enumerate(gpu_investigations or []):
        if profile.get("suite") != "affon_cell_gpu_variability":
            raise ValueError("unexpected GPU investigation report")
        for row in profile["runs"]:
            investigation_rows.append({"session": index + 1, "run": row["run"] + 1,
                "gpu_ms": row["gpu_command_median_ms"], "host_ms": row["host_total_median_ms"],
                "power": row["machine_state_before"]["power_source"],
                "thermal": row["machine_state_before"]["thermal_status"]})
    evidence = []
    checks = {
        "correctness": ("Does Affon return the right answer?", "A concrete case passed the independent output check."),
        "operator_peer": ("How fast is one operation?", "A matched Affon and peer timing report is linked."),
        "program_peer": ("How fast is the whole Program?", "A matched run includes Program overhead and synchronization."),
        "self_baseline": ("Is a regression reference ready?", "An accepted same-target Affon reference is linked."),
    }
    for key, (label, description) in checks.items():
        counts = totals["evidence"].get(key, {})
        linked = recorded.get(key, counts.get('linked', 0))
        evidence.append({"label": label, "description": description,
                         "linked": linked, "missing": totals['cells'] - linked})
    gaps = [row for row in cases if row["status"] == "candidate_gap"]
    mixed = [row for row in cases if row["status"] == "mixed_rounds"]
    better = [row for row in cases if row["status"] == "better_than_peer"]
    insights = [
        {"title": "Consistent candidate gaps", "body": f"{len(gaps)} cells exceed the provisional ratio in every round. The largest absolute difference is {gaps[0]['id']} at {(gaps[0]['affon_ms'] - gaps[0]['peer_ms']) * 1000:.1f} µs per call." if gaps else "No cell exceeds the provisional ratio in every round."},
        {"title": "Mixed measurements", "body": f"{len(mixed)} cells cross the threshold between rounds. Repeat on a controlled target before treating these as optimization targets."},
        {"title": "Consistently faster than PyTorch", "body": f"{len(better)} operator cells have Affon/PyTorch ratios below 1.0 in every round. These are diagnostic comparisons on the measured target."},
        {"title": "Checks still needed", "body": f"{totals['cells'] - recorded['program_peer']} Program peer cells and {totals['evidence']['self_baseline'].get('missing', 0)} baseline cells remain missing. Timing phase splits are diagnostic because accelerator work may finish during readback."},
    ]
    unstable = [row for row in baseline_calibrations if row["status"] == "Too variable"]
    if unstable:
        insights.append({"title": "Baseline timing is too variable", "body":
            f"{len(unstable)} calibration runs across {len({row['id'] for row in unstable})} cells exceeded the provisional repeatability limits. Keep baseline review open and stabilize measurement before making regression or optimization decisions."})
    boundaries = []
    if boundary_report is not None:
        if boundary_report.get("suite") != "affon_dispatch_boundary_audit":
            raise ValueError("unexpected boundary report suite")
        for row in boundary_report["boundaries"]:
            boundaries.append({"id": row["id"], "status": row["status"], "predicate": row["predicate"],
                               "below": " → ".join(row["below"]["path"]),
                               "at": " → ".join(row["at"]["path"]),
                               "below_shape": " × ".join(map(str, row["below"]["case"]["shape"])),
                               "at_shape": " × ".join(map(str, row["at"]["case"]["shape"])),
                               "below_hint": "dense input hint: on" if row["below"]["case"].get("dense_inputs") else "dense input hint: off",
                               "at_hint": "dense input hint: on" if row["at"]["case"].get("dense_inputs") else "dense input hint: off"})
        unobserved = boundary_report["summary"].get("unobserved_transition", 0)
        if unobserved:
            insights.append({"title": "Unobserved dispatch transitions", "body":
                f"{unobserved} declared planner transitions were masked by the compiler's selected path. They need a planner-level check or a workload that selects both paths before dispatch coverage is claimed."})
    phase_status = [
        {"name": "Foundation", "state": "Complete",
         "detail": "The saved results, report links, and baseline gate have executable checks."},
        {"name": "Coverage expansion", "state": "In progress" if active_batches else "Reviewed checkpoints",
         "detail": "Classify combinations and add concrete correctness, operator, Program, and dispatch checks by regime."},
        {"name": "Optimization", "state": "Diagnostic measurements" if cases else "Later",
         "detail": "Use general operation implementations and independent inputs; retain correctness and timing regressions."},
    ]
    distribution = [{"dimension": dimension, "rows": [
        {"name": name, **counts} for name, counts in current_distribution[dimension].items()]}
        for dimension in ("backend", "dtype", "family", "backend × dtype")] if "distribution" in totals else []
    return {"generated": datetime.now(ZoneInfo("Pacific/Auckland")).strftime("%d %b %Y, %H:%M NZ time"),
            "target": (next(iter({row['target'] for row in cases})) if len({row['target'] for row in cases}) == 1
                       else 'Multiple validation devices' if cases else 'No target'),
            "totalCells": totals["cells"], "probedCells": totals["evidence"]["correctness"].get("linked", 0),
            "recordedCorrectnessCells": recorded['correctness'], "recordedOperatorCells": recorded['operator_peer'],
            "gpuCandidateCells": sum(totals["distribution"]["backend"][name]["candidate"] for name in ("metal", "cuda")) if "distribution" in totals else 0,
            "peerCells": totals["evidence"]["operator_peer"].get("linked", 0),
            "acceptedBaselineCells": totals["evidence"]["self_baseline"].get("linked", 0),
            "unknownCells": totals["support"].get("unknown", 0),
            "measuredCases": len(cases), "boundaryChecks": len(boundaries),
            "observedBoundaries": sum(row["status"] == "expected_transition" for row in boundaries),
            "unobservedBoundaries": sum(row["status"] == "unobserved_transition" for row in boundaries),
            "cases": cases, "programCases": program_cases, "baselineCalibrations": baseline_calibrations,
            "activeBatches": [{k: v for k, v in batch.items() if k not in ('checks', 'directory')} for batch in active_batches or []],
            "baselineVerdicts": baseline_verdicts,
            "gpuProfiles": gpu_comparisons, "gpuInvestigationRuns": investigation_rows,
            "pairedGpuSessions": paired_gpu["sessions"] if paired_gpu else [],
            "pairedGpuChanges": paired_gpu["changes"] if paired_gpu else [],
            "evidence": evidence, "phaseStatus": phase_status, "distribution": distribution,
            "insights": insights, "boundaries": boundaries}


def render(payload, output, weiv=WEIV):
    if not weiv.is_file():
        raise FileNotFoundError(f"Weiv CLI missing: {weiv}")
    with tempfile.TemporaryDirectory(prefix="affon-weiv-report-") as folder:
        build = Path(folder)
        subprocess.run([str(weiv), "compile", str(ROOT / "test/benchmark/report/App.weiv"), str(build)], check=True)
        bundle = build / "report.mjs"
        subprocess.run(["bun", "build", str(build / "main.mjs"), "--target", "browser", "--format", "iife", "--outfile", str(bundle)],
                       check=True, capture_output=True, text=True)
        script = bundle.read_text().replace("</script", "<\\/script")
        props = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
        css = (build / "styles.css").read_text().replace("</style", "<\\/style")
        refresh = '<meta http-equiv="refresh" content="60">' if payload.get('activeBatches') else ''
        html = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                + refresh + "<title>Affon performance coverage</title><style>" + css + "</style></head><body>"
                "<div id=\"app\"></div><script>globalThis.__WEIV_PROPS__=" + props + "</script>"
                "<script>" + script + "</script></body></html>\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, prefix='.'+output.name+'-', delete=False) as temporary:
            temporary.write(html)
        Path(temporary.name).replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path,
                        default=ROOT / "test/benchmark/results/coverage-outcomes-current-mac-2026-10-29.json",
                        help="Coverage outcomes JSON; matrix is generated from current policy")
    parser.add_argument("--summary", type=Path, default=ROOT / "test/benchmark/results/cell-peer-current-summary-mac-2026-10-29.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "test/benchmark/results/cell-peer-current-mac-2026-10-29")
    parser.add_argument("--program-summary", type=Path, default=ROOT / "test/benchmark/results/cell-program-peer-summary-mac-2026-10-29.json")
    parser.add_argument("--program-raw-dir", type=Path, default=ROOT / "test/benchmark/results/cell-program-peer-mac-2026-10-29")
    parser.add_argument("--baseline-calibrations", type=Path, action="append", default=[
        ROOT / "test/benchmark/results/cell-baseline-calibration-mac-2026-10-29",
        ROOT / "test/benchmark/results/cell-baseline-calibration-long-mac-2026-10-29",
        ROOT / "test/benchmark/results/cell-baseline-phase-mac-2026-10-29",
        ROOT / "test/benchmark/results/cell-baseline-session-a-mac-2026-10-29",
        ROOT / "test/benchmark/results/cell-baseline-session-b-mac-2026-10-29"])
    parser.add_argument("--baseline-evaluations", type=Path,
                        default=ROOT / "test/benchmark/results/cell-baseline-evaluations-mac-2026-10-29")
    parser.add_argument("--gpu-variability", type=Path, action="append", default=None,
                        help="Session JSON; repeat for each comparable session")
    parser.add_argument("--gpu-investigation", type=Path, action="append", default=None)
    parser.add_argument("--paired-gpu", type=Path,
                        default=ROOT / "test/benchmark/results/paired-gpu-diagnostic-mac-2026-10-30.json")
    parser.add_argument("--boundaries", type=Path, default=ROOT / "test/benchmark/results/dispatch-boundaries-mac-2026-10-29.json")
    parser.add_argument("--output", type=Path, default=ROOT / "test/benchmark/results/performance-report.html")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("coverage_matrix", ROOT / "tools/coverage-matrix.py")
    matrix_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matrix_module)
    policy = json.loads((ROOT / "test/benchmark/coverage-policy.json").read_text())
    diagnostics = json.loads((ROOT / "test/benchmark/workload-coverage.json").read_text())
    outcomes = json.loads(args.outcomes.read_text())
    validate_evidence_artifacts(outcomes)
    matrix = matrix_module.build(policy, diagnostics, outcomes)
    boundary_report = json.loads(args.boundaries.read_text()) if args.boundaries.is_file() else None
    program_summary = json.loads(args.program_summary.read_text()) if args.program_summary.is_file() else None
    if matrix["summary"]["evidence"]["program_peer"].get("linked", 0) and program_summary is None:
        raise ValueError("linked Program evidence requires a Program summary")
    summary = json.loads(args.summary.read_text())
    validate_summary(summary, args.raw_dir, "operator_peer")
    if program_summary is not None:
        validate_summary(program_summary, args.program_raw_dir, "program_peer")
    gpu_paths = args.gpu_variability or [ROOT / "test/benchmark/results/cell-gpu-variability-session-a-mac-2026-10-29.json",
                                          ROOT / "test/benchmark/results/cell-gpu-variability-session-b-mac-2026-10-29.json"]
    gpu_profiles = [json.loads(path.read_text()) for path in gpu_paths] if all(path.is_file() for path in gpu_paths) else None
    if args.gpu_variability and gpu_profiles is None:
        raise FileNotFoundError("explicit GPU variability session missing")
    investigation_paths = args.gpu_investigation or [ROOT / f"test/benchmark/results/cell-gpu-variability-investigation-{index}-mac-2026-10-29.json"
                                                    for index in (1, 2, 3)]
    investigations = [json.loads(path.read_text()) for path in investigation_paths] if all(path.is_file() for path in investigation_paths) else None
    if args.gpu_investigation and investigations is None:
        raise FileNotFoundError("explicit GPU investigation session missing")
    paired_gpu = json.loads(args.paired_gpu.read_text()) if args.paired_gpu.is_file() else None
    if paired_gpu and paired_gpu.get("suite") != "affon_paired_gpu_diagnostic":
        raise ValueError("unexpected paired GPU diagnostic")
    batches = active_sweeps(ROOT / 'test/benchmark/results')
    completed = active_sweeps(ROOT / 'test/benchmark/results', completed=True)
    subprocess.run([sys.executable, str(ROOT / 'test/parity/index.py')], check=True)
    render(data(matrix, summary, args.raw_dir, boundary_report,
                program_summary, args.program_raw_dir, args.baseline_calibrations, gpu_profiles,
                args.baseline_evaluations, investigations, paired_gpu,
                batches, batches + completed), args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
