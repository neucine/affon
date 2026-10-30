import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "test/parity" / ("fixtures/" + name if name.endswith("_fixture.py") else name)
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchFacilityTest(unittest.TestCase):
    def test_contiguous_view_only_audit_tracks_materializing_path(self):
        audit = json.loads((ROOT / 'benchmarks/results/parity-metal-contiguous-view-only-audit-2026-10-31.json').read_text())
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        self.assertEqual(len(audit['decisions']), 2)
        for cell, decision in audit['decisions'].items():
            self.assertEqual(ledger[cell], decision)
            self.assertEqual(decision['support'], 'unsupported')
        for path, digest in audit['source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        probe = Path(audit['probe_artifact'])
        self.assertEqual(hashlib.sha256(probe.read_bytes()).hexdigest(), audit['probe_sha256'])

    def test_stat_boundary_review_retains_failures_and_complete_rerun(self):
        original = ROOT / 'benchmarks/results/parity-metal-stat-boundaries-256-2026-10-31'
        reviewed = ROOT / 'benchmarks/results/parity-metal-stat-peer-reviewed-4-2026-10-31'
        rows = json.loads((original / 'manifest.json').read_text())['cells']
        failed = {row['id'] for row in rows if row['operator_peer'] != 'passed' or
                  row['program_peer'] != 'passed'}
        self.assertEqual(len(rows), 256)
        self.assertEqual(len(failed), 4)
        self.assertTrue(all(cell.startswith('metal/bf16/variance_axis/outer_axis/') for cell in failed))
        rerun = json.loads((reviewed / 'manifest.json').read_text())['cells']
        self.assertEqual({row['id'] for row in rerun}, failed)
        for row in rerun:
            self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                             ('correct', 'passed', 'passed'))
            for kind in ('operator_peer', 'program_peer'):
                report = json.loads((reviewed / row[kind + '_artifact']).read_text())
                self.assertTrue(report['correctness']['passed'])
                self.assertEqual(len(report['affon_samples_ms']), 6)
                self.assertEqual(len(report['peer_samples_ms']), 6)
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        self.assertTrue(all(ledger[cell]['support'] == 'supported' for cell in failed))

    def test_scalar_reduce_and_invalid_scatter_keep_parity_and_boundaries(self):
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        for name, passing in (('parity-metal-scalar-reduce-to-shape-2026-10-31', 1),
                              ('parity-metal-invalid-scatter-2-2026-10-31', 1)):
            root = ROOT / 'benchmarks/results' / name
            rows = json.loads((root / 'manifest.json').read_text())['cells']
            self.assertEqual(sum(row['probe_status'] == 'correct' for row in rows), passing)
            for row in rows:
                if row['probe_status'] != 'correct':
                    self.assertEqual(ledger[row['id']]['support'], 'unsupported')
                    continue
                self.assertEqual((row['operator_peer'], row['program_peer']), ('passed', 'passed'))
                self.assertIn(row['id'], ledger)
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)
        audit = json.loads((ROOT / 'benchmarks/results/parity-metal-invalid-boundaries-2026-10-31.json').read_text())
        self.assertEqual(len(audit['decisions']), 3)
        for cell, decision in audit['decisions'].items():
            self.assertEqual(ledger[cell], decision)
        self.assertEqual(audit['affon']['metal/f32/scatter_add/invalid_index']['output'],
                         audit['peer']['metal/f32/scatter_add/invalid_index']['output'])
        for path, digest in audit['source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)

    def test_remaining_runnable_audit_retains_failures_and_selected_paths(self):
        root = ROOT / 'benchmarks/results/parity-metal-remaining-runnable-audit-25-2026-10-31'
        catalog = ROOT / 'test/parity/catalogs/metal-remaining-runnable-audit-25.json'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 25)
        failed = [row for row in manifest['cells'] if row['probe_status'] != 'correct']
        self.assertEqual(len(failed), 4)
        self.assertTrue(all('/variance_all/' in row['id'] or '/std_all/' in row['id'] for row in failed))
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        probes = {row['id']: row for row in json.loads((root / 'capability.json').read_text())['cases']}
        for row in manifest['cells']:
            cell = row['id']
            if cell.endswith('/view_only'):
                self.assertEqual(ledger[cell]['support'], 'unsupported')
                self.assertEqual(probes[cell]['selected_path'][-1]['implementation_id'], 'metal_contiguous')
            elif row['probe_status'] != 'correct':
                self.assertIn(cell, ledger)
                self.assertEqual(ledger[cell]['support'], 'supported')
                self.assertIn('parity-metal-variance-std-chunked-4-2026-10-31',
                              ledger[cell]['evidence']['correctness']['artifact'])
            else:
                self.assertEqual((row['operator_peer'], row['program_peer']), ('passed', 'passed'))
                self.assertIn(cell, ledger)
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)
                if '/offset' in cell:
                    self.assertEqual(probes[cell]['selected_path'][0]['implementation_id'], 'view.slice')
                if cell.endswith('/strided'):
                    self.assertIn('stride_2', json.loads((root / row['operator_peer_artifact']).read_text())['contract']['layout'])

    def test_corrected_variance_std_retains_complete_parity(self):
        root = ROOT / 'benchmarks/results/parity-metal-variance-std-chunked-4-2026-10-31'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(len(manifest['cells']), 4)
        for row in manifest['cells']:
            self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                             ('correct', 'passed', 'passed'))
            for kind in ('operator_peer', 'program_peer'):
                report = json.loads((root / row[kind + '_artifact']).read_text())
                self.assertTrue(report['correctness']['passed'])
                self.assertEqual(len(report['affon_samples_ms']), 15)
                self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_ragged_dot_and_contract_exclusions_are_reviewed(self):
        root = ROOT / 'benchmarks/results/parity-metal-ragged-dot-2-2026-10-31'
        manifest = json.loads((root / 'manifest.json').read_text())
        catalog = ROOT / 'test/parity/catalogs/metal-ragged-dot-2.json'
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 2)
        for row in manifest['cells']:
            self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                             ('correct', 'passed', 'passed'))
            for kind in ('operator_peer', 'program_peer'):
                report = json.loads((root / row[kind + '_artifact']).read_text())
                self.assertTrue(report['correctness']['passed'])
                self.assertEqual(len(report['affon_samples_ms']), 15)
                self.assertEqual(len(report['peer_samples_ms']), 15)
        audit = json.loads((ROOT / 'benchmarks/results/parity-metal-contract-exclusions-38-2026-10-31.json').read_text())
        self.assertEqual(len(audit['cells']), 38)
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        for row in audit['cells']:
            self.assertEqual(ledger[row['id']], {key: row[key] for key in ('support', 'reason')})
        for path, digest in audit['source_sha256'].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
        for contract in audit['contracts'].values():
            self.assertIn(contract['statement'], Path(contract['source']).read_text())

    def test_batched_index_selection_retains_numeric_and_timing_parity(self):
        root = ROOT / 'benchmarks/results/parity-metal-batched-index-select-130-2026-10-31'
        catalog = ROOT / 'test/parity/catalogs/metal-batched-index-select-130.json'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 130)
        for row in manifest['cells']:
            with self.subTest(cell=row['id']):
                self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                                 ('correct', 'passed', 'passed'))
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['contract']['input_shapes'][0]), 3)
                    self.assertEqual(len(report['contract']['input_shapes'][1]), 1)
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_batched_scatter_retains_numeric_and_timing_parity(self):
        root = ROOT / 'benchmarks/results/parity-metal-batched-scatter-130-2026-10-31'
        catalog = ROOT / 'test/parity/catalogs/metal-batched-scatter-130.json'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 130)
        for row in manifest['cells']:
            with self.subTest(cell=row['id']):
                self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                                 ('correct', 'passed', 'passed'))
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['contract']['input_shapes']), 3)
                    self.assertTrue(all(len(shape) == 3 for shape in report['contract']['input_shapes']))
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_batched_topk_retains_values_indices_and_timing(self):
        root = ROOT / 'benchmarks/results/parity-metal-batched-topk-130-2026-10-31'
        catalog = ROOT / 'test/parity/catalogs/metal-batched-topk-130.json'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 130)
        for row in manifest['cells']:
            with self.subTest(cell=row['id']):
                self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                                 ('correct', 'passed', 'passed'))
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertTrue(report['correctness']['output_hash_matches'])
                    self.assertEqual(len(report['contract']['input_shapes'][0]), 3)
                    self.assertEqual(len(report['contract']['outputs']), 2)
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_batched_dense_loss_retains_rank_three_numeric_and_timing_parity(self):
        root = ROOT / 'benchmarks/results/parity-metal-batched-dense-loss-132-reviewed-2026-10-31'
        catalog = ROOT / 'test/parity/catalogs/metal-batched-dense-loss-132.json'
        manifest = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest['catalog_sha256'])
        self.assertEqual(len(manifest['cells']), 132)
        for row in manifest['cells']:
            with self.subTest(cell=row['id']):
                self.assertEqual((row['probe_status'], row['operator_peer'], row['program_peer']),
                                 ('correct', 'passed', 'passed'))
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((root / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['contract']['input_shapes'][0]), 3)
                    self.assertEqual(report['contract']['output_shape'], [1])
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_canonical_gap_audit_retains_reruns_and_timing(self):
        root = ROOT / 'benchmarks/results/parity-metal-unknown-canonical-audit-2026-10-31'
        corrected = ROOT / 'benchmarks/results/parity-metal-unknown-canonical-topk-corrected-2026-10-31'
        rows = json.loads((root / 'manifest.json').read_text())['cells']
        rerun = json.loads((corrected / 'manifest.json').read_text())['cells']
        self.assertEqual((len(rows), len(rerun)), (36, 4))
        self.assertEqual({row['id'] for row in rows if row['probe_status'] != 'correct'},
                         {row['id'] for row in rerun})
        ledger = json.loads((ROOT / 'benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json').read_text())
        for batch, batch_rows in ((root, rows), (corrected, rerun)):
            for row in batch_rows:
                if row['probe_status'] != 'correct':
                    continue
                self.assertEqual((row['operator_peer'], row['program_peer']), ('passed', 'passed'))
                self.assertIn(row['id'], ledger)
                for kind in ('operator_peer', 'program_peer'):
                    report = json.loads((batch / row[kind + '_artifact']).read_text())
                    self.assertTrue(report['correctness']['passed'])
                    self.assertEqual(len(report['affon_samples_ms']), 15)
                    self.assertEqual(len(report['peer_samples_ms']), 15)

    def test_corrected_layer_norm_offset_contract_is_retained(self):
        root = ROOT / "benchmarks/results/parity-metal-layer-norm-views-128-offset-corrected-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["cells"]), 128)
        ledger = json.loads((ROOT / "benchmarks/results/coverage-outcomes-current-mac-2026-10-29.json").read_text())
        archive = ROOT / "benchmarks/results/cell-peer-before-layer-norm-offset-correction-2026-10-31"
        self.assertEqual(len(list((archive / "operator_peer").glob("*.json"))), 128)
        self.assertEqual(len(list((archive / "program_peer").glob("*.json"))), 128)
        for row in manifest["cells"]:
            self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                             ("correct", "passed", "passed"))
            if "/inner_offset_" in row["id"]:
                report = json.loads((root / row["operator_peer_artifact"]).read_text())
                self.assertEqual(report["contract"]["layout"], "offset_1_1")
                self.assertEqual(report["contract"]["input_shapes"][0],
                                 [int(row["id"].split("_r")[-1].split("_c")[0]) + 1,
                                  int(row["id"].split("_c")[-1]) + 1])
            self.assertEqual(ledger[row["id"]]["evidence"]["correctness"]["artifact"],
                             str((root / "capability.json").resolve()))

    def test_inner_reductions_retain_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-inner-reductions-256-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-inner-reductions-256.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 256)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)
                    if "inner_offset_" in row["id"]:
                        self.assertEqual(report["contract"]["layout"], "offset_1_1")

    def test_math_view_rerun_retains_exp_failures_and_complete_parity(self):
        first = ROOT / "benchmarks/results/parity-metal-math-views-192-2026-10-31"
        original = json.loads((first / "manifest.json").read_text())
        failed = [row for row in original["cells"] if row["operator_peer"] != "passed" or row["program_peer"] != "passed"]
        self.assertEqual(len(failed), 5)
        self.assertTrue(all("/f32/exp/" in row["id"] for row in failed))
        for row in failed:
            for kind in ("operator_peer", "program_peer"):
                self.assertFalse(json.loads((first / row[kind + "_artifact"]).read_text())["correctness"]["passed"])
        root = ROOT / "benchmarks/results/parity-metal-math-views-192-reviewed-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-math-views-192.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 192)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_transcendental_views_retain_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-transcendental-views-256-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-transcendental-views-256.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 256)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["dtype"], row["id"].split("/")[1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_layer_norm_view_rerun_retains_failures_and_complete_parity(self):
        first = ROOT / "benchmarks/results/parity-metal-layer-norm-views-128-2026-10-31"
        original = json.loads((first / "manifest.json").read_text())
        failed = [row for row in original["cells"] if row["operator_peer"] != "passed" or row["program_peer"] != "passed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["id"], "metal/f32/layer_norm/outer_axis/outer_offset_r3_c5")
        for kind in ("operator_peer", "program_peer"):
            self.assertFalse(json.loads((first / failed[0][kind + "_artifact"]).read_text())["correctness"]["passed"])
        root = ROOT / "benchmarks/results/parity-metal-layer-norm-views-128-reviewed-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-layer-norm-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_cast_views_retain_numeric_dtype_and_timing_parity(self):
        root = ROOT / "benchmarks/results/parity-metal-cast-views-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-cast-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                expected_dtype = "bf16" if row["id"].split("/")[1] == "f32" else "f32"
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["output_dtype"], expected_dtype)
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_matmul_lhs_views_retain_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-matmul-lhs-views-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-matmul-lhs-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        paths = set()
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)
                    if kind == "operator_peer":
                        paths.add(tuple(item["implementation_id"] for item in report["selected_path"]))
        self.assertEqual(paths, {("view.transpose", "metal.matmul.small"),
                                 ("view.slice", "metal.matmul.small"),
                                 ("view.transpose", "metal.matmul.tiled"),
                                 ("view.slice", "metal.matmul.tiled")})

    def test_comparison_views_retain_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-comparison-views-192-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-comparison-views-192.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 192)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["output_dtype"], "i64")
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_dense_matmul_grid_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-matmul-dense-grid-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-matmul-dense-grid-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_program_gap_rerun_retains_numeric_and_timing_parity(self):
        root = ROOT / "benchmarks/results/parity-metal-program-gaps-13-reviewed-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["cells"]), 13)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)
                    if kind == "program_peer" and "/batched" in row["id"]:
                        self.assertNotIn("metal.matmul_add.fused_tiled",
                                         [path["implementation_id"] for path in report["selected_path"]])

    def test_matmul_rhs_view_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-matmul-rhs-views-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-matmul-rhs-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["dtype"], row["id"].split("/")[1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_soft_loss_strided_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-soft-loss-strided-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-soft-loss-strided-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["output_shape"], [1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_index_select_view_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-index-select-views-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-index-select-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["dtype"], row["id"].split("/")[1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_scatter_strided_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-scatter-strided-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-scatter-strided-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["dtype"], row["id"].split("/")[1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_rms_norm_short_row_recheck_closes_timing_gap(self):
        root = ROOT / "benchmarks/results/parity-metal-rms-norm-r8-c7-recheck-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(len(manifest["cells"]), 1)
        row = manifest["cells"][0]
        self.assertEqual(row["id"], "metal/f32/rms_norm/short_row/r8_c7")
        self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                         ("correct", "passed", "passed"))
        for kind in ("operator_peer", "program_peer"):
            report = json.loads((root / row[kind + "_artifact"]).read_text())
            self.assertTrue(report["correctness"]["passed"])
            self.assertEqual(len(report["affon_samples_ms"]), 15)
            self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_rms_norm_view_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-rms-norm-views-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-rms-norm-views-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(report["contract"]["dtype"], row["id"].split("/")[1])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_topk_smallest_batch_checks_every_output_element(self):
        root = ROOT / "benchmarks/results/parity-metal-topk-smallest-192-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-topk-smallest-192.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 192)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertTrue(report["correctness"]["output_hash_matches"])
                    self.assertEqual([entry["dtype"] for entry in report["contract"]["outputs"]],
                                     [row["id"].split("/")[1], "i64"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_topk_hash_covers_every_value_and_index(self):
        fixture = load("topk_fixture.py")
        cell = {"id": "metal/bf16/topk/large/offset_axis1_r8_c7_k6_smallest",
                "dtype": "bf16", "shape": [8, 7], "axis": 1, "k": 6, "largest": False}
        values = fixture.reference(cell)
        changed = values.copy()
        changed[len(values) // 2 + 1] += 1
        self.assertNotEqual(fixture.output_hash(values), fixture.output_hash(changed))

    def test_topk_batch_retains_both_outputs_and_timings(self):
        root = ROOT / "benchmarks/results/parity-metal-topk-192-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-topk-192.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 192)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                dtype = row["id"].split("/")[1]
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertEqual(report["contract"]["outputs"][0]["dtype"], dtype)
                    self.assertEqual(report["contract"]["outputs"][1]["dtype"], "i64")
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertEqual(len(report["affon_samples_ms"]), 15)
                    self.assertEqual(len(report["peer_samples_ms"]), 15)

    def test_reviewed_binary_view_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-binary-views-256-reviewed-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-binary-views-256.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 256)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertEqual(report["cell_id"], row["id"])
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertGreater(len(report["affon_samples_ms"]), 0)
                    self.assertGreater(len(report["peer_samples_ms"]), 0)

    def test_binary_view_batch_retains_probe_failures_and_complete_passes(self):
        root = ROOT / "benchmarks/results/parity-metal-binary-views-256-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-binary-views-256.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 256)
        failures = 0
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                if row["id"].startswith("metal/f32/") and "/offset/" in row["id"]:
                    failures += 1
                    self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                     ("incorrect", "probe_not_correct", "probe_not_correct"))
                    continue
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertEqual(report["cell_id"], row["id"])
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertGreater(len(report["affon_samples_ms"]), 0)
                    self.assertGreater(len(report["peer_samples_ms"]), 0)
        self.assertEqual(failures, 64)

    def test_nonlinear_view_batch_retains_complete_parity_evidence(self):
        root = ROOT / "benchmarks/results/parity-metal-nonlinear-views-256-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-nonlinear-views-256.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 256)
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                 ("correct", "passed", "passed"))
                for kind in ("operator_peer", "program_peer"):
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertEqual(report["cell_id"], row["id"])
                    self.assertTrue(report["correctness"]["passed"])
                    self.assertGreater(len(report["affon_samples_ms"]), 0)
                    self.assertGreater(len(report["peer_samples_ms"]), 0)

    def test_outer_stat_index_batch_retains_single_peer_failure(self):
        root = ROOT / "benchmarks/results/parity-metal-outer-stat-index-128-2026-10-31"
        manifest = json.loads((root / "manifest.json").read_text())
        catalog = ROOT / "test/parity/catalogs/metal-outer-stat-index-128.json"
        self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
        self.assertEqual(len(manifest["cells"]), 128)
        failed = "metal/bf16/std_axis/outer_axis/strided_r3_c129"
        for row in manifest["cells"]:
            with self.subTest(cell=row["id"]):
                self.assertEqual(row["probe_status"], "correct")
                expected = "incorrect" if row["id"] == failed else "passed"
                for kind in ("operator_peer", "program_peer"):
                    self.assertEqual(row[kind], expected)
                    report = json.loads((root / row[kind + "_artifact"]).read_text())
                    self.assertEqual(report["cell_id"], row["id"])
                    self.assertEqual(report["correctness"]["passed"], expected == "passed")
                    self.assertGreater(len(report["affon_samples_ms"]), 0)
                    self.assertGreater(len(report["peer_samples_ms"]), 0)

    def test_reviewed_metal_batches_retain_complete_parity_evidence(self):
        batches = {
            "parity-metal-shapes-128-2026-10-31": "metal-matmul-indexed-shapes-128.json",
            "parity-metal-dot-widths-128-reviewed-2026-10-31": "metal-dot-widths-128.json",
            "parity-metal-dot-layouts-128-2026-10-31": "metal-dot-layouts-128.json",
            "parity-metal-one-hot-shapes-128-2026-10-31": "metal-one-hot-shapes-128.json",
            "parity-metal-one-hot-layouts-128-2026-10-31": "metal-one-hot-layouts-128.json",
            "parity-metal-one-hot-invalid-index-128-2026-10-31": "metal-one-hot-invalid-index-128.json",
            "parity-metal-one-hot-batched-128-2026-10-31": "metal-one-hot-batched-128.json",
            "parity-metal-indexed-loss-grid-128-2026-10-31": "metal-indexed-loss-grid-128.json",
            "parity-metal-quantized-grid-128-2026-10-31": "metal-quantized-grid-128.json",
            "parity-metal-quantized-views-128-2026-10-31": "metal-quantized-views-128.json",
            "parity-metal-where-layouts-128-2026-10-31": "metal-where-layouts-128.json",
            "parity-metal-gather-views-128-2026-10-31": "metal-gather-views-128.json",
            "parity-metal-scatter-add-128-2026-10-31": "metal-scatter-add-128.json",
            "parity-metal-dense-loss-128-2026-10-31": "metal-dense-loss-128.json",
            "parity-metal-log-softmax-nll-128-2026-10-31": "metal-log-softmax-nll-128.json",
            "parity-metal-masked-fill-128-2026-10-31": "metal-masked-fill-128.json",
            "parity-metal-reduce-to-shape-128-2026-10-31": "metal-reduce-to-shape-128.json",
            "parity-metal-indexed-loss-views-128-2026-10-31": "metal-indexed-loss-views-128.json",
            "parity-metal-arg-axis-views-128-2026-10-31": "metal-arg-axis-views-128.json",
            "parity-metal-softmax-views-128-2026-10-31": "metal-softmax-views-128.json",
            "parity-metal-variance-views-128-2026-10-31": "metal-variance-views-128.json",
            "parity-metal-outer-softmax-views-128-2026-10-31": "metal-outer-softmax-views-128.json",
            "parity-metal-outer-reductions-128-2026-10-31": "metal-outer-reductions-128.json",
            "parity-metal-outer-stat-index-128-reviewed-2026-10-31": "metal-outer-stat-index-128.json",
        }
        for run_name, catalog_name in batches.items():
            with self.subTest(run=run_name):
                root = ROOT / "benchmarks/results" / run_name
                manifest = json.loads((root / "manifest.json").read_text())
                catalog = ROOT / "test/parity/catalogs" / catalog_name
                self.assertEqual(hashlib.sha256(catalog.read_bytes()).hexdigest(), manifest["catalog_sha256"])
                self.assertEqual(len(manifest["cells"]), 128)
                for row in manifest["cells"]:
                    with self.subTest(cell=row["id"]):
                        self.assertEqual((row["probe_status"], row["operator_peer"], row["program_peer"]),
                                         ("correct", "passed", "passed"))
                        for kind in ("operator_peer", "program_peer"):
                            artifact = root / row[kind + "_artifact"]
                            self.assertTrue(artifact.is_file())
                            report = json.loads(artifact.read_text())
                            self.assertEqual(report["cell_id"], row["id"])
                            self.assertTrue(report["correctness"]["passed"])
                            self.assertGreater(len(report["affon_samples_ms"]), 0)
                            self.assertGreater(len(report["peer_samples_ms"]), 0)

    def test_report_counts_all_cells_and_shows_failed_timing(self):
        report = load("report.py")
        manifest = {"target": "GPU/Linux/x86_64", "catalog_sha256": "a" * 64,
                    "binary_sha256": "b" * 64, "iterations": 2, "samples": 3,
                    "rounds": 2, "cells": [
                        {"id": "cuda/f32/add/small_dense", "probe_status": "correct",
                         "operator_peer": "passed", "operator_peer_artifact": "operator_peer/a.json",
                         "operator_peer_affon_ms": 1.5, "operator_peer_peer_ms": 1.0,
                         "program_peer": "incorrect", "program_peer_artifact": "program_peer_failures/a.json",
                         "program_peer_affon_ms": 2.0, "program_peer_peer_ms": 1.0},
                        {"id": "cuda/f32/cos/small_dense", "probe_status": "probe_failed",
                         "operator_peer": "probe_not_correct", "program_peer": "probe_not_correct"}]}
        output = report.render(manifest)
        self.assertIn("Cases run: 2", output)
        self.assertIn("Numerical check passed: 1", output)
        self.assertIn("Program failures: 1", output)
        self.assertIn("2 / 1", output)
        self.assertIn("probe_not_correct", output)
        self.assertIn("[incorrect](program_peer_failures/a.json)", output)

    def test_rerun_refuses_existing_evidence_directory(self):
        runner = load("run.py")
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(
                catalog=ROOT / "test/parity/catalogs/cuda-first-256.json",
                binary=Path(__file__), output_dir=Path(directory),
                iterations=2, samples=3, rounds=2)
            with self.assertRaisesRegex(FileExistsError, "fresh output directory"):
                runner.run(args)


if __name__ == "__main__":
    unittest.main()
