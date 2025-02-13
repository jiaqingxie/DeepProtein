import os
import sys
import argparse
import torch
import wandb


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.ProteinPred as models
from DeepProtein.utils import get_hf_model_embedding

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Prediction with DeepProtein")
    parser.add_argument('--target_encoding', type=str, default='CNN', help='Encoding method for target proteins')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_proj', type=str, default='your_project_name', help='wandb project name')
    parser.add_argument('--lr', type=float, default=0.0001, help='0.0001/0.00001')
    parser.add_argument('--epochs', type=int, default=100, help='50/100')
    parser.add_argument('--compute_pos_enc', type=bool, default=False, help='compute position encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    target_encoding = args.target_encoding
    wandb_project = args.wandb_proj
    lr = args.lr
    epochs = args.epochs
    compute_pos = args.compute_pos_enc
    batch_size = args.batch_size

    job_name = f"Stability + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()

    #  Test on Beta lactamase
    train_stab = Stability(path + '/DeepProtein/data', 'train')
    valid_stab = Stability(path + '/DeepProtein/data', 'valid')
    test_stab = Stability(path + '/DeepProtein/data', 'test')

    if target_encoding in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
        train_protein_processed, train_target, train_protein_idx = collate_fn(train_stab, graph=True)
        valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_stab, graph=True)
        test_protein_processed, test_target, test_protein_idx = collate_fn(test_stab, graph=True)

    else:
        train_protein_processed, train_target, train_protein_idx = collate_fn(train_stab)
        valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_stab)
        test_protein_processed, test_target, test_protein_idx = collate_fn(test_stab)





    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[1024, 1024],
                             train_epoch=epochs,
                             LR=lr,
                             batch_size=batch_size,
                             )
    config['multi'] = False
    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)
    model.LLM_test_and_log(test_protein_processed, test_target, "stability", False)



