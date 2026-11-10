import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("performance_report", ROOT / "tools/render-performance-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class PerformanceReportTest(unittest.TestCase):
    def test_timing_tables_include_valid_checkpoints_without_replacing_reviewed_records(self):
        fixture = ROOT / 'test/benchmark/results/parity-cuda-sweep-harness-2026-11-05'
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            batch = results / 'parity-timings'
            shutil.copytree(fixture, batch)
            pending = next((batch/'program_peer').glob('*.json'))
            pending_cell = json.loads(pending.read_text())['cell_id']
            pending.unlink()
            state = json.loads((batch/'progress.json').read_text())
            state['requested_phase_finished'] = False
            (batch/'progress.json').write_text(json.dumps(state))
            batches = report.active_sweeps(results)
            ids = {row['id'] for row in json.loads((batch/'catalog.json').read_text())['cases']}
            rows, _ = report.checkpoint_measurements(batches, ids)
            self.assertEqual({row['id'] for row in rows['operator_peer']}, ids)
            self.assertEqual({row['id'] for row in rows['program_peer']}, ids - {pending_cell})
            old = {**rows['operator_peer'][0], 'ratio': 123, 'evidence_state': 'Reviewed record'}
            retained = {**old, 'id': 'metal/f32/retained/small'}
            merged = report.merge_measurements([old, retained], rows['operator_peer'])
            self.assertEqual(len(merged), len(ids)+1)
            replacement = next(row for row in merged if row['id'] == old['id'])
            self.assertEqual(replacement['ratio'], 123)
            self.assertEqual(replacement['evidence_state'], 'Reviewed record')
            path = next((batch/'operator_peer').glob('*.json'))
            raw = json.loads(path.read_text())
            raw['rounds'][0]['output_hash_matches'] = False
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                report.checkpoint_measurements(batches, ids)

    def test_finished_sweep_keeps_recorded_checks_after_leaving_active_panel(self):
        fixture = ROOT / 'test/benchmark/results/parity-cuda-sweep-harness-2026-11-05'
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            batch = results / 'parity-transition'
            shutil.copytree(fixture, batch)
            state = json.loads((batch/'progress.json').read_text())
            state['requested_phase_finished'] = False
            (batch/'progress.json').write_text(json.dumps(state))
            active = report.active_sweeps(results)
            self.assertEqual(len(active), 1)
            self.assertEqual(report.active_sweeps(results, completed=True), [])
            state['requested_phase_finished'] = True
            (batch/'progress.json').write_text(json.dumps(state))
            self.assertEqual(report.active_sweeps(results), [])
            completed = report.active_sweeps(results, completed=True)
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]['checks'], active[0]['checks'])

    def test_live_coverage_unions_records_without_double_counting_or_promoting(self):
        kinds = ('correctness', 'operator_peer', 'program_peer')
        cells = []
        for cell, linked in [('cuda/f32/add/small', kinds[:2]), ('cuda/f32/add/large', ()),
                             ('cuda/f32/multiply/small', ()), ('metal/f32/add/small', kinds[:1])]:
            cells.append({'id': cell, 'family': 'binary',
                          'evidence': {kind: {'artifact': 'retained.json'} if kind in linked else 'missing' for kind in kinds}})
        groups = {'backend': {'cuda': 3, 'metal': 1}, 'dtype': {'f32': 4},
                  'family': {'binary': 4}, 'backend × dtype': {'cuda / f32': 3, 'metal / f32': 1}}
        matrix = {'cells': cells, 'summary': {'distribution': {
            dimension: {name: {'candidate': count, 'supported': 0} for name, count in values.items()}
            for dimension, values in groups.items()}}}
        before = copy.deepcopy(matrix)
        batches = [{'checks': {'correctness': ['cuda/f32/add/small', 'cuda/f32/add/large'],
                               'operator_peer': ['cuda/f32/add/large'], 'program_peer': ['cuda/f32/add/large']}},
                   {'checks': {'correctness': ['cuda/f32/add/large'], 'operator_peer': ['cuda/f32/add/large']}}]
        counts, distribution = report.recorded_coverage(matrix, batches)
        self.assertEqual(counts, {'correctness': 3, 'operator_peer': 2, 'program_peer': 1})
        self.assertEqual(distribution['backend']['cuda']['correctness_recorded'], 2)
        self.assertEqual(distribution['backend']['cuda']['operator_peer_recorded'], 2)
        self.assertEqual(distribution['backend']['cuda']['candidate'], 3)
        self.assertEqual(distribution['backend']['cuda']['supported'], 0)
        self.assertEqual(distribution['family']['binary']['correctness_recorded'], 3)
        self.assertEqual(matrix, before)
        with self.assertRaisesRegex(ValueError, 'unknown coverage'):
            report.recorded_coverage(matrix, [{'checks': {'correctness': ['cuda/f32/unknown/small']}}])

    def test_evidence_artifact_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.json"
            path.write_text("first")
            outcomes = {"cpu/f32/add/small_dense": {"evidence": {"correctness": {
                "artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}}
            report.validate_evidence_artifacts(outcomes)
            path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "evidence artifact changed"):
                report.validate_evidence_artifacts(outcomes)

    def test_measured_rows_require_the_linked_raw_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            expected = Path(folder) / "expected.json"
            other = Path(folder) / "other.json"
            matrix = {"cells": [{"id": "cpu/f32/add/small_dense", "evidence": {
                "operator_peer": {"artifact": str(expected)}}}]}
            cases = [{"id": "cpu/f32/add/small_dense"}]
            with self.assertRaisesRegex(ValueError, "different raw artifact"):
                report.validate_measured_links(matrix, cases, {cases[0]["id"]: other}, "operator_peer")

    def test_summary_must_match_raw_rounds(self):
        base = ROOT / "test/benchmark/results"
        summary = json.loads((base / "cell-peer-current-summary-mac-2026-10-29.json").read_text())
        raw = base / "cell-peer-current-mac-2026-10-29"
        report.validate_summary(summary, raw, "operator_peer")
        summary["cases"][0]["ratio"] += 0.1
        with self.assertRaisesRegex(ValueError, "summary differs from raw"):
            report.validate_summary(summary, raw, "operator_peer")

    def test_gpu_sessions_show_between_session_drift(self):
        paths = [ROOT / f"test/benchmark/results/cell-gpu-variability-session-{name}-mac-2026-10-29.json"
                 for name in ("a", "b")]
        profiles = [json.loads(path.read_text()) for path in paths]
        matrix = {"suite": "affon_coverage_matrix", "summary": {"cells": 0,
                  "support": {}, "evidence": {key: {} for key in
                  ("correctness", "operator_peer", "program_peer", "self_baseline")}}}
        summary = {"suite": "affon_cell_diagnostic_summary", "cases": []}
        with tempfile.TemporaryDirectory() as folder:
            result = report.data(matrix, summary, Path(folder), gpu_profiles=profiles)
        comparison = result["gpuProfiles"][0]
        self.assertAlmostEqual(comparison["gpu_ratio"], 4.36, places=2)
        self.assertAlmostEqual(comparison["host_ratio"], 2.79, places=2)
        self.assertEqual(comparison["sessions"], 2)

    def test_report_data_comes_from_matrix_and_raw_cases(self):
        matrix = {"suite": "affon_coverage_matrix", "summary": {"cells": 20,
                  "support": {"supported": 2, "unknown": 18},
                  "evidence": {"correctness": {"linked": 2, "missing": 18},
                               "operator_peer": {"linked": 1, "missing": 19},
                               "program_peer": {"missing": 20}, "self_baseline": {"missing": 20}}}}
        summary = {"suite": "affon_cell_diagnostic_summary", "cases": [
            {"id": "cpu/f32/add/small_dense", "status": "candidate_gap", "ratio": 2,
             "affon_ms": 0.02, "peer_ms": 0.01, "round_ratios": [2, 2]}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "case.json"
            path.write_text(json.dumps({"suite": "affon_cell_benchmark", "kind": "operator_peer", "cell_id": "cpu/f32/add/small_dense",
                                        "target": "local", "contract": {"shape": [2, 4]},
                                        "selected_path": [{"implementation_id": "cpu_add"}]}))
            result = report.data(matrix, summary, Path(folder))
            self.assertEqual((result["totalCells"], result["probedCells"], result["peerCells"]), (20, 2, 1))
            self.assertEqual((result["measuredCases"], result["boundaryChecks"]), (1, 0))
            self.assertEqual(result["cases"][0]["path"], "cpu_add")
            self.assertEqual(result["evidence"][0]["label"], "Does Affon return the right answer?")
            self.assertEqual(result["evidence"][3]["label"], "Is a regression reference ready?")

    def test_saved_html_mounts_and_filters_from_file_url(self):
        html = ROOT / "test/benchmark/results/performance-report.html"
        results = ROOT / "test/benchmark/results"
        operator_count = len(json.loads((results / "cell-peer-current-summary-mac-2026-10-29.json").read_text())["cases"])
        program_count = len(json.loads((results / "cell-program-peer-summary-mac-2026-10-29.json").read_text())["cases"])
        outcomes = json.loads((results / "coverage-outcomes-current-mac-2026-10-29.json").read_text())
        correctness_count = sum("correctness" in row.get("evidence", {}) for row in outcomes.values())
        matrix_spec = importlib.util.spec_from_file_location('current_matrix', ROOT / 'tools/coverage-matrix.py')
        matrix_module = importlib.util.module_from_spec(matrix_spec)
        matrix_spec.loader.exec_module(matrix_module)
        matrix = matrix_module.build(json.loads((ROOT / 'test/benchmark/coverage-policy.json').read_text()),
                                     json.loads((ROOT / 'test/benchmark/workload-coverage.json').read_text()), outcomes)
        expected = matrix['summary']
        script = r'''
const fs=require('fs');
const {JSDOM}=require(process.argv[2]);
const operatorCount=Number(process.argv[3]), programCount=Number(process.argv[4]), correctnessCount=Number(process.argv[5]);
const expected=JSON.parse(process.argv[6]);
const errors=[];
const dom=new JSDOM(fs.readFileSync(process.argv[1],'utf8'),{runScripts:'dangerously',url:'file://' + process.argv[1],beforeParse(w){w.addEventListener('error',e=>errors.push(e.message))}});
const d=dom.window.document;
const saved=dom.window.__WEIV_PROPS__;
const comparisonRows=()=>d.querySelector('.operator-table').querySelectorAll('tbody tr').length;
if(errors.length || comparisonRows()!==100 || d.querySelector('.program-table').querySelectorAll('tbody tr').length!==100 || d.querySelectorAll('.distribution article').length!==4 || d.querySelectorAll('.boundary-table tbody tr').length!==6) process.exit(1);
const groups=[...d.querySelectorAll('.distribution article')].map(x=>x.textContent.replace(/\s+/g,''));
for (const [index,dimension] of ['backend','dtype','family','backend × dtype'].entries()) {
  for (const [name,counts] of Object.entries(expected.distribution[dimension])) {
    if(!groups[index].includes(name.replace(/\s+/g,'')+counts.candidate)) process.exit(14);
  }
}
if(!d.body.textContent.includes('GPU candidate cells')) process.exit(15);
for (const [index,group] of saved.distribution.entries()) {
  for (const row of group.rows) {
    if(!groups[index].includes(row.name.replace(/\s+/g,'')+row.candidate+row.correctness_recorded+row.operator_peer_recorded)) process.exit(27);
  }
}
const framework=d.querySelector('.framework');
if(!framework || !framework.textContent.includes(expected.cells+' candidate cells') || !framework.textContent.includes(saved.measuredCases+' operator comparisons') || !framework.textContent.includes('6 pairs')) process.exit(4);
if(saved.measuredCases!==saved.cases.length || saved.measuredCases!==saved.recordedOperatorCells || saved.peerCells!==operatorCount) process.exit(29);
if(saved.programCases.length!==saved.evidence[2].linked) process.exit(30);
const promotedReduction='cuda/f32/reduce_to_shape/long_axis/dense_r128_c129';
if(saved.cases.find(x=>x.id===promotedReduction)?.evidence_state!=='Reviewed record' ||
   saved.programCases.find(x=>x.id===promotedReduction)?.evidence_state!=='Reviewed record') process.exit(34);
const phases=d.querySelector('.phases');
if(!phases || !phases.textContent.includes('Foundation') || !phases.textContent.includes('Coverage expansion') || !phases.textContent.includes('Optimization')) process.exit(12);
if(!d.querySelector('.metrics.aggregate').textContent.includes(String(operatorCount)) || !d.querySelector('.metrics.aggregate').textContent.includes(String(saved.recordedOperatorCells)) || !d.querySelector('.metrics.aggregate').textContent.includes(String(saved.recordedCorrectnessCells))) process.exit(13);
for(const check of saved.evidence) {
  if(!d.querySelector('.evidence').textContent.includes(check.linked+' of '+expected.cells+' cells')) process.exit(28);
}
const active=dom.window.__WEIV_PROPS__.activeBatches;
if(d.querySelectorAll('.active-validation').length!==active.length) process.exit(25);
for(const [index,batch] of active.entries()) {
  const panel=d.querySelectorAll('.active-validation')[index];
  const values=[...panel.querySelectorAll('.metrics strong')].map(x=>Number(x.textContent));
  if(JSON.stringify(values)!==JSON.stringify([batch.total,batch.numerical,batch.operator,batch.program]) || !panel.textContent.includes('pending')) process.exit(26);
}
const checklist=d.querySelector('.evidence');
if(!checklist || !checklist.textContent.includes('Does Affon return the right answer?') || !checklist.textContent.includes('Is a regression reference ready?')) process.exit(5);
if(!d.body.textContent.includes('matmul → add_bias') || !d.body.textContent.includes('Too variable')) process.exit(7);
if(!d.body.textContent.includes('Drift between 2 sessions') || !d.body.textContent.includes('4.36×') || !d.body.textContent.includes('2.79×') || !d.body.textContent.includes('Accepted baseline cells: 0')) process.exit(8);
if(!d.body.textContent.includes('Independent session gate') || !d.body.textContent.includes('1.86×') || !d.body.textContent.includes('1.73×') || !d.body.textContent.includes('2.03×')) process.exit(9);
if(!d.body.textContent.includes('Focused GPU timing investigation') || !d.body.textContent.includes('353.4 µs')) process.exit(10);
if(!d.body.textContent.includes('Alternating cell check') || !d.body.textContent.includes('1.04×') || !d.body.textContent.includes('1.02×')) process.exit(11);
const boundaries=d.querySelector('.boundary-table');
if(!boundaries || !boundaries.textContent.includes('dense input hint: on') || !boundaries.textContent.includes('Same implementation selected')) process.exit(6);
const tableRows=kind=>[...d.querySelector(`.${kind}-table tbody`).querySelectorAll('tr')];
const flush=()=>new Promise(resolve=>dom.window.setTimeout(resolve,0));
const choose=async(kind,name,value)=>{const el=d.querySelector(`[aria-label="${kind} filters"] [data-filter="${name}"]`);el.value=value;el.dispatchEvent(new dom.window.Event('change',{bubbles:true}));await flush();};
const search=async(kind,value)=>{const el=d.querySelector(`[aria-label="${kind} filters"] [data-filter="query"]`);el.value=value;el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));await flush();};
(async()=>{
for(const kind of ['Operator','Program']) {
  const rows=()=>tableRows(kind.toLowerCase()==='operator'?'operator':'program');
  const label=d.querySelector(`[data-pager="${kind.toLowerCase()}"] span`);
  const allRows=kind==='Operator'?saved.cases:saved.programCases;
  if(!label.textContent.includes('of '+allRows.length+' matches') || rows().length!==100) process.exit(16);
  await choose(kind,'device','metal'); await choose(kind,'dtype','bf16');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/'))) process.exit(17);
  await choose(kind,'operation','add');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/add/'))) process.exit(18);
  await choose(kind,'status','candidate_gap');
  if(rows().some(row=>!row.textContent.includes('Candidate gap'))) process.exit(19);
  await choose(kind,'status','all'); await choose(kind,'operation','all'); await choose(kind,'dtype','all'); await choose(kind,'device','all');
  await search(kind,'metal/bf16/add');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/add'))) process.exit(20);
  await search(kind,'');
  if(rows().length!==100) process.exit(21);
  const first=rows()[0].textContent;
  d.querySelector(`[data-pager="${kind.toLowerCase()}"] button:last-child`).click();
  await flush();
  if(rows().length!==100 || rows()[0].textContent===first || !label.textContent.includes('101–200')) process.exit(22);
  await choose(kind,'device','cuda');
  if(!label.textContent.startsWith('1–100') || rows().some(row=>!row.textContent.includes('cuda/'))) process.exit(23);
  const cudaRows=allRows.filter(row=>row.id.startsWith('cuda/'));
  if(!label.textContent.includes('of '+cudaRows.length+' matches')) process.exit(31);
  await choose(kind,'device','all');
}
if(errors.length) process.exit(24);
})().catch(error=>{console.error(error);process.exit(35)});
'''
        jsdom = ROOT.parent / "weiv/node_modules/jsdom"
        if html.is_file() and jsdom.is_dir():
            subprocess.run(["node", "-e", script, str(html), str(jsdom),
                            str(operator_count), str(program_count), str(correctness_count), json.dumps(expected)],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
