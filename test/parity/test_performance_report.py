import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("performance_report", ROOT / "tools/render-performance-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class PerformanceReportTest(unittest.TestCase):
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
        base = ROOT / "benchmarks/results"
        summary = json.loads((base / "cell-peer-current-summary-mac-2026-10-29.json").read_text())
        raw = base / "cell-peer-current-mac-2026-10-29"
        report.validate_summary(summary, raw, "operator_peer")
        summary["cases"][0]["ratio"] += 0.1
        with self.assertRaisesRegex(ValueError, "summary differs from raw"):
            report.validate_summary(summary, raw, "operator_peer")

    def test_gpu_sessions_show_between_session_drift(self):
        paths = [ROOT / f"benchmarks/results/cell-gpu-variability-session-{name}-mac-2026-10-29.json"
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
        html = ROOT / "benchmarks/results/performance-report.html"
        results = ROOT / "benchmarks/results"
        operator_count = len(json.loads((results / "cell-peer-current-summary-mac-2026-10-29.json").read_text())["cases"])
        program_count = len(json.loads((results / "cell-program-peer-summary-mac-2026-10-29.json").read_text())["cases"])
        outcomes = json.loads((results / "coverage-outcomes-current-mac-2026-10-29.json").read_text())
        correctness_count = sum("correctness" in row.get("evidence", {}) for row in outcomes.values())
        script = r'''
const fs=require('fs');
const {JSDOM}=require(process.argv[2]);
const operatorCount=Number(process.argv[3]), programCount=Number(process.argv[4]), correctnessCount=Number(process.argv[5]);
const errors=[];
const dom=new JSDOM(fs.readFileSync(process.argv[1],'utf8'),{runScripts:'dangerously',url:'file://' + process.argv[1],beforeParse(w){w.addEventListener('error',e=>errors.push(e.message))}});
const d=dom.window.document;
const comparisonRows=()=>d.querySelector('.operator-table').querySelectorAll('tbody tr').length;
if(errors.length || comparisonRows()!==100 || d.querySelector('.program-table').querySelectorAll('tbody tr').length!==100 || d.querySelectorAll('.distribution article').length!==4 || d.querySelectorAll('.boundary-table tbody tr').length!==6) process.exit(1);
const groups=[...d.querySelectorAll('.distribution article')].map(x=>x.textContent.replace(/\s+/g,''));
if(!groups[0].includes('metal13240') || !groups[0].includes('cuda2011') || !groups[1].includes('bf166504') || !groups[1].includes('f3271396')) process.exit(14);
if(!groups[3].includes('metal/bf165700') || !d.body.textContent.includes('GPU candidate cells')) process.exit(15);
const framework=d.querySelector('.framework');
if(!framework || !framework.textContent.includes('17261 candidate cells') || !framework.textContent.includes(operatorCount+' operator comparisons') || !framework.textContent.includes('6 pairs')) process.exit(4);
const phases=d.querySelector('.phases');
if(!phases || !phases.textContent.includes('Foundation') || !phases.textContent.includes('Coverage expansion') || !phases.textContent.includes('Optimization')) process.exit(12);
if(!d.querySelector('.metrics').textContent.includes(String(operatorCount)) || !d.querySelector('.evidence').textContent.includes(correctnessCount+' of 17261 cells') || !d.querySelector('.evidence').textContent.includes(programCount+' of 17261 cells')) process.exit(13);
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
const choose=(kind,name,value)=>{const el=d.querySelector(`[aria-label="${kind} filters"] [data-filter="${name}"]`);el.value=value;el.dispatchEvent(new dom.window.Event('change',{bubbles:true}));};
const search=(kind,value)=>{const el=d.querySelector(`[aria-label="${kind} filters"] [data-filter="query"]`);el.value=value;el.dispatchEvent(new dom.window.Event('input',{bubbles:true}));};
for(const kind of ['Operator','Program']) {
  const rows=()=>tableRows(kind.toLowerCase()==='operator'?'operator':'program');
  const label=d.querySelector(`[data-pager="${kind.toLowerCase()}"] span`);
  if(!label.textContent.includes('of 12') || rows().length!==100) process.exit(16);
  choose(kind,'device','metal'); choose(kind,'dtype','bf16');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/'))) process.exit(17);
  choose(kind,'operation','add');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/add/'))) process.exit(18);
  choose(kind,'status','candidate_gap');
  if(rows().some(row=>!row.textContent.includes('Candidate gap'))) process.exit(19);
  choose(kind,'status','all'); choose(kind,'operation','all'); choose(kind,'dtype','all'); choose(kind,'device','all');
  search(kind,'metal/bf16/add');
  if(rows().length===0 || rows().some(row=>!row.textContent.includes('metal/bf16/add'))) process.exit(20);
  search(kind,'');
  if(rows().length!==100) process.exit(21);
  const first=rows()[0].textContent;
  d.querySelector(`[data-pager="${kind.toLowerCase()}"] button:last-child`).click();
  if(rows().length!==100 || rows()[0].textContent===first || !label.textContent.includes('101–200')) process.exit(22);
  choose(kind,'device','cuda');
  if(!label.textContent.startsWith('1–100') || rows().some(row=>!row.textContent.includes('cuda/'))) process.exit(23);
  choose(kind,'device','all');
}
if(errors.length) process.exit(24);
'''
        jsdom = ROOT.parent / "weiv/node_modules/jsdom"
        if html.is_file() and jsdom.is_dir():
            subprocess.run(["node", "-e", script, str(html), str(jsdom),
                            str(operator_count), str(program_count), str(correctness_count)],
                           check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
