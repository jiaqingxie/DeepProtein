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
import DeepProtein.ProteinPred as models
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop, CRISPROutcome

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

    job_name = f"CRISPR + {target_encoding}"
    wandb.init(project=wandb_project, name=job_name)
    wandb.config.update(args)

    path = os.getcwd()



    label_list = retrieve_label_name_list('Leenay')

    data = CRISPROutcome(name='Leenay', label_name=label_list[0])
    split = data.get_split()

    train_GuideSeq, y_train = list(split['train']['GuideSeq']), list(split['train']['Y'])
    val_GuideSeq, y_valid = list(split['valid']['GuideSeq']), list(split['valid']['Y'])
    test_GuideSeq, y_test = list(split['test']['GuideSeq']), list(split['test']['Y'])

    # print(y_train)
    train_CRISPR = list(zip(train_GuideSeq, y_train))
    valid_CRISPR = list(zip(val_GuideSeq, y_valid))
    test_CRISPR = list(zip(test_GuideSeq, y_test))

    if target_encoding in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT',
                           'Graphormer']:

        train_protein_1, train_target, train_protein_idx = collate_fn(train_CRISPR, graph=True,
                                                                                           unsqueeze=False)
        valid_protein_1, valid_target, valid_protein_idx = collate_fn(valid_CRISPR, graph=True,
                                                                                           unsqueeze=False)
        test_protein_1, test_target, test_protein_idx = collate_fn(test_CRISPR, graph=True,
                                                                                       unsqueeze=False)

    else:

        train_protein_1, train_target, train_protein_idx = collate_fn(train_CRISPR, graph=False,
                                                                                           unsqueeze=True)
        valid_protein_1, valid_target, valid_protein_idx = collate_fn(valid_CRISPR, graph=False,
                                                                                           unsqueeze=True)
        test_protein_1, test_target, test_protein_idx = collate_fn(test_CRISPR, graph=False,
                                                                                       unsqueeze=True)

    if target_encoding == "prot_bert":
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        embedding_model = BertModel.from_pretrained("Rostlab/prot_bert").to("cuda")
        train_protein_1 = get_hf_model_embedding(train_protein_1, tokenizer, embedding_model,
                                                         target_encoding)
        valid_protein_1 = get_hf_model_embedding(valid_protein_1, tokenizer, embedding_model,
                                                         target_encoding)
        test_protein_1 = get_hf_model_embedding(test_protein_1, tokenizer, embedding_model,
                                                        target_encoding)

    elif target_encoding == "esm_1b":
        from transformers import EsmTokenizer, EsmModel

        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        embedding_model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to("cuda")
        train_protein_1 = get_hf_model_embedding(train_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        valid_protein_1 = get_hf_model_embedding(valid_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        test_protein_1 = get_hf_model_embedding(test_protein_1, tokenizer, embedding_model,
                                                target_encoding)

    elif target_encoding == "esm_2":
        from transformers import EsmTokenizer, EsmModel

        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")
        train_protein_1 = get_hf_model_embedding(train_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        valid_protein_1 = get_hf_model_embedding(valid_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        test_protein_1 = get_hf_model_embedding(test_protein_1, tokenizer, embedding_model,
                                                target_encoding)

    elif target_encoding == "prot_t5":
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        embedding_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to("cuda")
        train_protein_1 = get_hf_model_embedding(train_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        valid_protein_1 = get_hf_model_embedding(valid_protein_1, tokenizer, embedding_model,
                                                 target_encoding)
        test_protein_1 = get_hf_model_embedding(test_protein_1, tokenizer, embedding_model,
                                                target_encoding)



    train, _, _ = data_process(X_target=train_protein_1, y=train_target,
                               target_encoding=target_encoding,
                               split_method='random', frac=[0.99998, 1e-5, 1e-5],
                               random_seed=1)
    _, val, _ = data_process(X_target=valid_protein_1, y=valid_target,
                             target_encoding=target_encoding,
                             split_method='random', frac=[1e-5, 0.99998, 1e-5],
                             random_seed=1)

    _, _, test = data_process(X_target=test_protein_1, y=test_target,
                              target_encoding=target_encoding,
                              split_method='random', frac=[1e-5, 1e-5, 0.99998],
                              random_seed=1)

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


