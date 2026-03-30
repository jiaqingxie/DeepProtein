import tempfile
import unittest
from pathlib import Path

import wandb

import DeepProtein.PPI as ppi_models
import DeepProtein.ProteinPred as protein_models
from DeepProtein.load_dataset import load_pair_dataset, load_single_dataset
from DeepProtein.utils import (
    data_process_PPI_loader,
    data_process_loader_Protein_Prediction,
    generate_config,
)


class PyGPositionalEncodingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]

    def _write_custom_single_file(self, file_path):
        file_path.write_text(
            "\n".join(
                [
                    "ACDEFGHIK,0.1",
                    "LMNPQRSTV,0.2",
                    "MKTAYIAKQ,0.3",
                    "RQGIRKAFV,0.4",
                    "GAVLIPFWY,0.5",
                    "STNQCMHKR,0.6",
                    "AAAAACCCC,0.7",
                    "GGGGGTTTT,0.8",
                    "VVVVVVVVV,0.9",
                    "YYYYYYYYY,1.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_custom_pair_file(self, file_path):
        file_path.write_text(
            "\n".join(
                [
                    "ACDEFGHIK,LMNPQRSTV,0.1",
                    "MKTAYIAKQ,RQGIRKAFV,0.2",
                    "GAVLIPFWY,STNQCMHKR,0.3",
                    "AAAAACCCC,GGGGGTTTT,0.4",
                    "VVVVVVVVV,YYYYYYYYY,0.5",
                    "MNPQRSTVW,ACDEFGHIK,0.6",
                    "LMNPQRSTV,AAAAACCCC,0.7",
                    "STNQCMHKR,VVVVVVVVV,0.8",
                    "RQGIRKAFV,GAVLIPFWY,0.9",
                    "YYYYYYYYY,MKTAYIAKQ,1.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_single_protein_train_with_compute_pos_enc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "single.csv"
            self._write_custom_single_file(data_file)

            train, val, test = load_single_dataset(
                "Custom",
                str(self.repo_root),
                "PyG_GCN",
                your_file=str(data_file),
            )
            config = generate_config(
                target_encoding="PyG_GCN",
                cls_hidden_dims=[16],
                train_epoch=1,
                LR=1e-5,
                batch_size=2,
                result_folder=str(Path(tmpdir) / "single_results"),
            )
            config["binary"] = False
            config["multi"] = False
            config["num_workers"] = 0

            dataset = data_process_loader_Protein_Prediction(
                train.index.values,
                train.Label.values,
                train,
                **{**config, "compute_pos_enc": True},
            )
            graph, _label = dataset[0]
            self.assertTrue(hasattr(graph, "pe"))
            self.assertEqual(graph.pe.shape[1], config["pyg_pos_enc_dim"])

            run = wandb.init(mode="disabled", reinit=True)
            try:
                model = protein_models.model_initialize(**config)
                model.train(train, val, test, verbose=False, compute_pos_enc=True)
                preds = model.predict(test)
            finally:
                wandb.finish()

            self.assertIsNotNone(run)
            self.assertEqual(len(preds), len(test))

    def test_ppi_train_with_compute_pos_enc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "pair.csv"
            self._write_custom_pair_file(data_file)

            train, val, test = load_pair_dataset(
                "Custom",
                str(self.repo_root),
                "PyG_GCN",
                your_file=str(data_file),
            )
            config = generate_config(
                target_encoding="PyG_GCN",
                cls_hidden_dims=[16],
                train_epoch=1,
                LR=1e-5,
                batch_size=2,
                result_folder=str(Path(tmpdir) / "ppi_results"),
            )
            config["binary"] = False
            config["multi"] = False
            config["num_workers"] = 0

            dataset = data_process_PPI_loader(
                train.index.values,
                train.Label.values,
                train,
                **{**config, "compute_pos_enc": True},
            )
            graph_1, graph_2, _label = dataset[0]
            self.assertTrue(hasattr(graph_1, "pe"))
            self.assertTrue(hasattr(graph_2, "pe"))
            self.assertEqual(graph_1.pe.shape[1], config["pyg_pos_enc_dim"])
            self.assertEqual(graph_2.pe.shape[1], config["pyg_pos_enc_dim"])

            run = wandb.init(mode="disabled", reinit=True)
            try:
                model = ppi_models.model_initialize(**config)
                model.train(train, val, test, verbose=False, compute_pos_enc=True)
                preds = model.predict(test)
            finally:
                wandb.finish()

            self.assertIsNotNone(run)
            self.assertEqual(len(preds), len(test))


if __name__ == "__main__":
    unittest.main()
