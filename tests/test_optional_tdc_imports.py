import builtins
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class OptionalTdcImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _write_custom_pair_file(self, file_path):
        file_path.write_text(
            "\n".join(
                [
                    "ACDEFGHIK,LMNPQRSTV,1",
                    "MKTAYIAKQ,RQGIRKAFV,0",
                    "GAVLIPFWY,STNQCMHKR,1",
                    "AAAAACCCC,GGGGGTTTT,0",
                    "VVVVVVVVV,YYYYYYYYY,1",
                    "MNPQRSTVW,ACDEFGHIK,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _import_load_dataset_without_tdc(self):
        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tdc" or name.startswith("tdc."):
                raise ModuleNotFoundError("No module named 'tdc'")
            return original_import(name, globals, locals, fromlist, level)

        for module_name in list(sys.modules):
            if module_name == "DeepProtein.load_dataset" or module_name == "tdc" or module_name.startswith("tdc."):
                sys.modules.pop(module_name, None)

        patcher = mock.patch("builtins.__import__", new=blocked_import)
        patcher.start()
        self.addCleanup(patcher.stop)

        module = importlib.import_module("DeepProtein.load_dataset")
        self.addCleanup(sys.modules.pop, "DeepProtein.load_dataset", None)
        return module

    def test_non_tdc_pair_path_imports_without_tdc(self):
        load_dataset = self._import_load_dataset_without_tdc()
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_file = Path(tmpdir) / "ppi_pairs.csv"
            self._write_custom_pair_file(pair_file)
            train, val, test = load_dataset.load_pair_dataset(
                "Custom",
                str(self.repo_root),
                "PyG_GCN",
                your_file=str(pair_file),
            )
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)

    def test_tdc_dataset_still_requires_tdc(self):
        load_dataset = self._import_load_dataset_without_tdc()
        with self.assertRaises(ModuleNotFoundError):
            load_dataset.load_pair_dataset("TAP", str(self.repo_root), "PyG_GAT")


if __name__ == "__main__":
    unittest.main()
