"""Guard the permanent parity layout and its navigable artifact index."""
import importlib.util
import json
from pathlib import Path
import py_compile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PARITY = ROOT / 'test/parity'


class ParityLayoutTest(unittest.TestCase):
    def test_catalog_generators_compile_and_target_catalog_directory(self):
        generators = sorted((PARITY / 'catalogs/generators').glob('generate-*.py'))
        self.assertGreater(len(generators), 40)
        for path in generators:
            py_compile.compile(str(path), doraise=True)
            source = path.read_text()
            self.assertNotIn('parents[3]', source)
            self.assertNotIn('Path(__file__).with_name(', source)

    def test_index_covers_catalogs_and_retained_manifests(self):
        spec = importlib.util.spec_from_file_location('parity_index', PARITY / 'index.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        index = (ROOT / 'test/benchmarks/results/PARITY-INDEX.md').read_text()
        self.assertEqual(index, module.render())
        self.assertEqual(len(module.catalog_rows()), len(list((PARITY / 'catalogs').glob('*.json'))))
        for batch, count, complete in module.batch_rows():
            self.assertIn(batch.name, index)
            self.assertLessEqual(complete, count)
            self.assertEqual(count, len(json.loads((batch / 'manifest.json').read_text())['cells']))


if __name__ == '__main__':
    unittest.main()
