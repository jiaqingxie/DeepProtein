import os
import sys
import argparse
import torch
import wandb

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepPurpose_PP.dataset import *
import DeepPurpose_PP.utils as utils
import DeepPurpose_PP.ProteinPred as models


def parse_args():
    parser = argparse.ArgumentParser(description="Protein Prediction with DeepPurpose++")
    parser.add_argument('--target_encoding', type=str, default='CNN', help='Encoding method for target proteins')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--wandb_proj', type=str, default='your_project_name', help='wandb project name')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    target_encoding = args.target_encoding
    wandb_project = args.wandb_proj
    job_name = f"Solubility + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()
 


    train_fluo = Solubility(path + '/DeepPurpose_PP/data', 'train')
    valid_fluo = Solubility(path + '/DeepPurpose_PP/data', 'valid')
    test_fluo = Solubility(path + '/DeepPurpose_PP/data', 'test')

    if target_encoding in ['DGL_GCN']:
        train_protein_processed, train_target, train_protein_idx = collate_fn(train_fluo, graph=True)
        valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_fluo, graph=True)
        test_protein_processed, test_target, test_protein_idx = collate_fn(test_fluo, graph=True)

    else:
        train_protein_processed, train_target, train_protein_idx = collate_fn(train_fluo)
        valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_fluo)
        test_protein_processed, test_target, test_protein_idx = collate_fn(test_fluo)

    train, _, _ = utils.data_process(X_target=train_protein_processed, y=train_target, target_encoding=target_encoding,
                                     # drug_encoding= drug_encoding,
                                     split_method='random', frac=[0.99998, 1e-5, 1e-5],
                                     random_seed=1)

    _, val, _ = utils.data_process(X_target=valid_protein_processed, y=valid_target, target_encoding=target_encoding,
                                   # drug_encoding= drug_encoding,
                                   split_method='random', frac=[1e-5, 0.99998, 1e-5],
                                   random_seed=1)

    _, _, test = utils.data_process(X_target=test_protein_processed, y=test_target, target_encoding=target_encoding,
                                    # drug_encoding= drug_encoding,
                                    split_method='random', frac=[1e-5, 1e-5, 0.99998],
                                    random_seed=1)

    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[1024, 1024],
                             train_epoch=100,
                             LR=0.0001,
                             batch_size=32,
                             )
    
    config['binary'] = True

    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)
    model.train(train, val, test)

