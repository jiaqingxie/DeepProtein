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
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Prediction with ProB")
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

    job_name = f"SAbDab_Chen + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()

    data = Develop(name='SAbDab_Chen')
    split = data.get_split()

    train_antibody_1, train_antibody_2 = to_two_seq(split, 'train', 'Antibody', sep=",")
    valid_antibody_1, valid_antibody_2 = to_two_seq(split, 'valid', 'Antibody', sep=",")
    test_antibody_1, test_antibody_2 = to_two_seq(split, 'test', 'Antibody', sep=",")

    y_train, y_valid, y_test = split['train']['Y'], split['valid']['Y'], split['test']['Y']

    train_TAP = list(zip(train_antibody_1, train_antibody_2, y_train))
    valid_TAP = list(zip(valid_antibody_1, valid_antibody_2, y_valid))
    test_TAP = list(zip(test_antibody_1, test_antibody_2, y_test))

    if target_encoding in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT',
                           'Graphormer']:

        train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train_TAP, graph=True,
                                                                                           unsqueeze=False)
        valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid_TAP, graph=True,
                                                                                           unsqueeze=False)
        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_TAP, graph=True,
                                                                                       unsqueeze=False)

    else:

        train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train_TAP, graph=False,
                                                                                           unsqueeze=False)
        valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid_TAP, graph=False,
                                                                                           unsqueeze=False)
        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_TAP, graph=False,
                                                                                       unsqueeze=False)
    train, _, _ = data_process(X_target=train_protein_1, X_target_=train_protein_2, y=train_target,
                               target_encoding=target_encoding,
                               split_method='random', frac=[0.99998, 1e-5, 1e-5],
                               random_seed=1)
    _, val, _ = data_process(X_target=valid_protein_1, X_target_=valid_protein_2, y=valid_target,
                             target_encoding=target_encoding,
                             split_method='random', frac=[1e-5, 0.99998, 1e-5],
                             random_seed=1)

    _, _, test = data_process(X_target=test_protein_1, X_target_=test_protein_2, y=test_target,
                              target_encoding=target_encoding,
                              split_method='random', frac=[1e-5, 1e-5, 0.99998],
                              random_seed=1)
    #
    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[512],
                             train_epoch=epochs,
                             LR=lr,
                             batch_size=batch_size,
                             )

    config['binary'] = False
    config['multi'] = False
    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)
    model.train(train, val, test)

