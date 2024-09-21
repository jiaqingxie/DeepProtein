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
import ProB.TokenPred as models

from tdc.single_pred import Epitope, Paratope

def parse_args():
    parser = argparse.ArgumentParser(description="Protein Prediction with ProB")
    parser.add_argument('--target_encoding', type=str, default='Token_CNN', help='Encoding method for target proteins')
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

    job_name = f"SAbDab_Liberis + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()

    data_class, name, X = Paratope, 'SAbDab_Liberis', 'Antibody'
    data = data_class(name=name)
    split = data.get_split()
    train_data, valid_data, test_data = split['train'], split['valid'], split['test']
    vocab_set = set()

    train_vocab, train_positive_ratio = data2vocab(train_data, train_data, X)
    valid_vocab, valid_positive_ratio = data2vocab(valid_data, train_data, X)
    test_vocab, test_positive_ratio = data2vocab(test_data, train_data, X)

    vocab_set = train_vocab.union(valid_vocab)
    vocab_set = vocab_set.union(test_vocab)
    vocab_lst = list(vocab_set)

    train_data = standardize_data(train_data, vocab_lst, X)
    valid_data = standardize_data(valid_data, vocab_lst, X)
    test_data = standardize_data(test_data, vocab_lst, X)

    train_set = data_process_loader_Token_Protein_Prediction(train_data)
    valid_set = data_process_loader_Token_Protein_Prediction(valid_data)
    test_set = data_process_loader_Token_Protein_Prediction(test_data)
    #
    # if target_encoding in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP',  'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
    #     train_protein_processed, train_target, train_protein_idx = collate_fn(train_IMDB, graph=True)
    #     valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_IMDB, graph=True)
    #     test_protein_processed, test_target, test_protein_idx = collate_fn(test_IMDB, graph=True)
    #
    # else:
    #     train_protein_processed, train_target, train_protein_idx = collate_fn(train_IMDB)
    #     valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_IMDB)
    #     test_protein_processed, test_target, test_protein_idx = collate_fn(test_IMDB)
    #
    # train, _, _ = utils.data_process(X_target=train_protein_processed, y=train_target, target_encoding=target_encoding,
    #                                  # drug_encoding= drug_encoding,
    #                                  split_method='random', frac=[0.99998, 1e-5, 1e-5],
    #                                  random_seed=1)
    #
    # _, val, _ = utils.data_process(X_target=valid_protein_processed, y=valid_target, target_encoding=target_encoding,
    #                                # drug_encoding= drug_encoding,
    #                                split_method='random', frac=[1e-5, 0.99998, 1e-5],
    #                                random_seed=1)
    #
    # _, _, test = utils.data_process(X_target=test_protein_processed, y=test_target, target_encoding=target_encoding,
    #                                 # drug_encoding= drug_encoding,
    #                                 split_method='random', frac=[1e-5, 1e-5, 0.99998],
    #                                 random_seed=1)

    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[1024, 1024],
                             train_epoch=epochs,
                             LR=lr,
                             batch_size=batch_size ,
                             )
    config['multi'] = False
    config['binary'] = True
    config['token'] = True
    config['in_channels'] = 20
    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)
    model.train(train_set, valid_set, test_set, batch_size=batch_size)


