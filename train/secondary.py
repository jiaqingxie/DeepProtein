import os
import sys
import argparse
import torch
import wandb


module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.TokenPred as models
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

    job_name = f"Secondary + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()

    #  Test on Secondary Structure
    train_data = SecondaryStructure(path + '/DeepProtein/data', 'train')
    valid_data = SecondaryStructure(path + '/DeepProtein/data', 'valid')
    test_data = SecondaryStructure(path + '/DeepProtein/data', 'cb513')

    train_vocab, train_positive_ratio = data2vocab_dataset(train_data)
    valid_vocab, valid_positive_ratio = data2vocab_dataset(valid_data)
    test_vocab, test_positive_ratio = data2vocab_dataset(test_data)

    vocab_set = train_vocab.union(valid_vocab)
    vocab_set = vocab_set.union(test_vocab)
    vocab_lst = list(vocab_set)

    train_data = standardize_data_dataset(train_data, vocab_lst)
    valid_data = standardize_data_dataset(valid_data, vocab_lst)
    test_data = standardize_data_dataset(test_data, vocab_lst)

    config = generate_config(target_encoding=target_encoding,
                             cls_hidden_dims=[1024, 1024],
                             train_epoch=epochs,
                             LR=lr,
                             batch_size=batch_size ,
                             )
    config['multi'] = True
    config['token'] = True
    config['binary'] = False
    config['classes'] = 3
    config['in_channels'] = 21
    torch.manual_seed(args.seed)
    model = models.model_initialize(**config)

    train = data_process_loader_Token_Protein_Prediction(train_data)
    valid = data_process_loader_Token_Protein_Prediction(valid_data)
    test = data_process_loader_Token_Protein_Prediction(test_data)
    model.train(train, valid, test, batch_size=batch_size)


