import unittest

import torch
import torch.nn.functional as F

from DeepProtein.encoders import (
    PyG_ChebNet,
    PyG_GAT,
    PyG_GCN,
    PyG_GIN,
    PyG_GraphSAGE,
    PyG_TAGConv,
)
from DeepProtein.utils import ATOM_FDIM, BOND_FDIM


try:
    from torch_geometric.data import Batch, Data
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent skip
    raise unittest.SkipTest("torch-geometric is required for PyG encoder smoke tests") from exc


class PyGEncoderSmokeTests(unittest.TestCase):
    def _make_batch(self):
        graphs = []
        for node_count in (3, 4):
            edge_pairs = []
            for idx in range(node_count - 1):
                edge_pairs.extend([[idx, idx + 1], [idx + 1, idx]])

            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.zeros((edge_index.shape[1], BOND_FDIM), dtype=torch.float32)
            graphs.append(
                Data(
                    x=torch.randn(node_count, ATOM_FDIM, dtype=torch.float32),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
            )
        return Batch.from_data_list(graphs)

    def test_all_pyg_encoders_forward(self):
        batch = self._make_batch()
        encoder_specs = {
            "PyG_GCN": PyG_GCN(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
            ),
            "PyG_GAT": PyG_GAT(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
                heads=2,
            ),
            "PyG_GraphSAGE": PyG_GraphSAGE(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
            ),
            "PyG_GIN": PyG_GIN(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
            ),
            "PyG_ChebNet": PyG_ChebNet(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
                cheb_k=3,
            ),
            "PyG_TAGConv": PyG_TAGConv(
                in_feats=ATOM_FDIM,
                hidden_feats=[8, 8],
                activation=F.relu,
                predictor_dim=16,
            ),
        }

        for name, encoder in encoder_specs.items():
            with self.subTest(encoder=name):
                output = encoder(batch)
                self.assertEqual(output.shape, (2, 16))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
