import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils import data

import DeepProtein.PPI as ppi_models
from DeepProtein.load_dataset import load_pair_dataset
from DeepProtein.utils import (
    PYG_TARGET_ENCODINGS,
    data_process_PPI_loader,
    generate_config,
    pyg_ppi_collate_func,
)


class PyGPPIPathSmokeTests(unittest.TestCase):
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

    def test_all_pyg_pair_encoders_forward_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_file = Path(tmpdir) / "ppi_pairs.csv"
            self._write_custom_pair_file(pair_file)

            for method in sorted(PYG_TARGET_ENCODINGS):
                with self.subTest(encoder=method):
                    train, _val, test = load_pair_dataset(
                        "Custom",
                        str(self.repo_root),
                        method,
                        your_file=str(pair_file),
                    )
                    config = generate_config(
                        target_encoding=method,
                        cls_hidden_dims=[16],
                        train_epoch=1,
                        LR=1e-5,
                        batch_size=2,
                        result_folder=str(Path(tmpdir) / method),
                    )
                    config["binary"] = True
                    config["multi"] = False
                    config["num_workers"] = 0

                    model = ppi_models.model_initialize(**config)
                    dataset = data_process_PPI_loader(
                        train.index.values,
                        train.Label.values,
                        train,
                        **config,
                    )
                    loader = data.DataLoader(
                        dataset,
                        batch_size=2,
                        shuffle=False,
                        collate_fn=pyg_ppi_collate_func,
                    )

                    v_d, v_p, label = next(iter(loader))
                    v_d = v_d.to(model.device)
                    v_p = v_p.to(model.device)
                    label = label.float().to(model.device).view(-1)
                    score = model.model(v_d, v_p)
                    loss = torch.nn.BCELoss()(torch.sigmoid(score).view(-1), label)
                    self.assertEqual(score.shape, (2, 1))
                    self.assertTrue(torch.isfinite(loss))

                    preds = model.predict(test)
                    self.assertEqual(len(preds), len(test))


if __name__ == "__main__":
    unittest.main()
