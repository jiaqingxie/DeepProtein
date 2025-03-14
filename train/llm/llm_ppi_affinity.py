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
import DeepProtein.PPI as models

def parse_args():
    parser = argparse.ArgumentParser(description="PPI Prediction with `DeepProtein`")
    parser.add_argument('--target_encoding', type=str, default='CNN', help='Encoding method for target proteins')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_proj', type=str, default='your_project_name', help='wandb project name')
    parser.add_argument('--lr', type=float, default=0.0001, help='0.0001/0.00001')
    parser.add_argument('--epochs', type=int, default=100, help='50/100')
    parser.add_argument('--compute_pos_enc', type=bool, default=False, help='compute position encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    target_encoding = args.target_encoding
    wandb_project = args.wandb_proj
    lr = args.lr
    epochs = args.epochs
    compute_pos = args.compute_pos_enc
    batch_size = args.batch_size

    job_name = f"PPI_Affinity + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()


    test_ppi = PPI_Affinity(path + '/DeepProtein/data', 'test')

    if target_encoding in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:

        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_ppi, graph=True, unsqueeze= False)

    else:
        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_ppi, graph=False, unsqueeze= False)





    #
    config = generate_config(target_encoding = target_encoding,
                             cls_hidden_dims = [512],
                             train_epoch = epochs,
                             LR = lr,
                             batch_size = batch_size,
                            )
    config['binary'] = False
    config['multi'] = False
    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)
    model.LLM_test_and_log(test_protein_1, test_protein_2, test_target, "ppi_affinity")
