import os
import sys
import argparse
import torch
import wandb

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ProB.dataset import *
import ProB.utils as utils
import ProB.PPI as models

def parse_args():
    parser = argparse.ArgumentParser(description="PPI Prediction with `ProB`")
    parser.add_argument('--target_encoding', type=str, default='CNN', help='Encoding method for target proteins')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_proj', type=str, default='your_project_name', help='wandb project name')
    parser.add_argument('--lr', type=float, default=0.0001, help='0.0001/0.00001')
    parser.add_argument('--epochs', type=int, default=100, help='50/100')
    parser.add_argument('--compute_pos_enc', type=bool, default=False, help='compute position encoding')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')


    return parser.parse_args()
if __name__ == '__main__':
    path = os.getcwd()
    train_ppi = PPI_Affinity(path + '/ProB/data', 'train')

    train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train_ppi, graph=False)



    target_encoding = 'CNN'
    train, val, test = data_process(X_target = train_protein_1, X_target_ = train_protein_2, y = train_target,
                    target_encoding = target_encoding,
                    split_method='random',
                    random_seed = 1)
    #
    config = generate_config(target_encoding = target_encoding,
                             cls_hidden_dims = [512],
                             train_epoch = 20,
                             LR = 0.001,
                             batch_size = 128,
                            )

    model = models.model_initialize(**config)
    model.train(train, val, test)