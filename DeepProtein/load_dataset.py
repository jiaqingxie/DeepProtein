import os, sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop, CRISPROutcome
from tdc.single_pred import Epitope, Paratope
from DeepProtein.dataset import *
import DeepProtein.utils as utils

def load_single_dataset(dataset_name, path, method):
    # loading single
    if dataset_name == "Beta":
        train = Beta_lactamase(path + '/DeepProtein/data', 'train')
        valid = Beta_lactamase(path + '/DeepProtein/data', 'valid')
        test = Beta_lactamase(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Fluorescence":
        train = FluorescenceDataset(path + '/DeepProtein/data', 'train')
        valid = FluorescenceDataset(path + '/DeepProtein/data', 'valid')
        test = FluorescenceDataset(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Solubility":
        train = Solubility(path + '/DeepProtein/data', 'train')
        valid = Solubility(path + '/DeepProtein/data', 'valid')
        test = Solubility(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Stability":
        train = Stability(path + '/DeepProtein/data', 'train')
        valid = Stability(path + '/DeepProtein/data', 'valid')
        test = Stability(path + '/DeepProtein/data', 'test')
    elif dataset_name == "SubCellular":
        train = Subcellular(path + '/DeepProtein/data', 'train')
        valid = Subcellular(path + '/DeepProtein/data', 'valid')
        test = Subcellular(path + '/DeepProtein/data', 'test')
    elif dataset_name == "SubCellular_Binary":
        train = BinarySubcellular(path + '/DeepProtein/data', 'train')
        valid = BinarySubcellular(path + '/DeepProtein/data', 'valid')
        test = BinarySubcellular(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Fold":
        train = Fold(path + '/DeepProtein/data', 'train')
        valid = Fold(path + '/DeepProtein/data', 'valid')
        test = Fold(path + '/DeepProtein/data', 'test_superfamily_holdout')
    elif dataset_name == "CRISPR":
        label_list = retrieve_label_name_list('Leenay')
        data = CRISPROutcome(name='Leenay', label_name=label_list[0])
        split = data.get_split()
        train_GuideSeq, y_train = list(split['train']['GuideSeq']), list(split['train']['Y'])
        val_GuideSeq, y_valid = list(split['valid']['GuideSeq']), list(split['valid']['Y'])
        test_GuideSeq, y_test = list(split['test']['GuideSeq']), list(split['test']['Y'])
        train = list(zip(train_GuideSeq, y_train))
        valid = list(zip(val_GuideSeq, y_valid))
        test = list(zip(test_GuideSeq, y_test))

    #### deal with targeting (sequence-based and structure-based):

    if method in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP',  'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
        if dataset_name in ["CRISPR", "Stability"]:
            train_protein_processed, train_target, train_protein_idx = collate_fn(train, graph=True, unsqueeze=False)
            valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid, graph=True, unsqueeze=False)
            test_protein_processed, test_target, test_protein_idx = collate_fn(test, graph=True, unsqueeze=False)
        else:
            train_protein_processed, train_target, train_protein_idx = collate_fn(train, graph=True)
            valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid, graph=True)
            test_protein_processed, test_target, test_protein_idx = collate_fn(test, graph=True)

    else:
        if dataset_name in ["Stability"]:
            train_protein_processed, train_target, train_protein_idx = collate_fn(train, unsqueeze=False)
            valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid, unsqueeze=False)
            test_protein_processed, test_target, test_protein_idx = collate_fn(test, unsqueeze=False)
        else:
            train_protein_processed, train_target, train_protein_idx = collate_fn(train)
            valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid)
            test_protein_processed, test_target, test_protein_idx = collate_fn(test)


    # deal with pretrained protein language models
    tokenizer, embedding_model = None, None
    if method == "prot_bert":
        from transformers import BertModel, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        embedding_model = BertModel.from_pretrained("Rostlab/prot_bert").to("cuda")

    elif method == "esm_1b":
        from transformers import EsmTokenizer, EsmModel
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        embedding_model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to("cuda")

    elif method == "esm_2":
        from transformers import EsmTokenizer, EsmModel
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")

    elif method == "prot_t5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        embedding_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to("cuda")

    if method in ["prot_bert", "esm_1b", "esm_2", "prot_t5"]:
        train_protein_processed = get_hf_model_embedding(train_protein_processed, tokenizer, embedding_model,
                                                         method)
        valid_protein_processed = get_hf_model_embedding(valid_protein_processed, tokenizer, embedding_model,
                                                         method)
        test_protein_processed = get_hf_model_embedding(test_protein_processed, tokenizer, embedding_model,
                                                        method)

    # process train, val and test
    train, _, _ = utils.data_process(X_target=train_protein_processed, y=train_target, target_encoding=method,
                                     # drug_encoding= drug_encoding,
                                     split_method='random', frac=[0.99998, 1e-5, 1e-5],
                                     random_seed=1)

    _, val, _ = utils.data_process(X_target=valid_protein_processed, y=valid_target, target_encoding=method,
                                   # drug_encoding= drug_encoding,
                                   split_method='random', frac=[1e-5, 0.99998, 1e-5],
                                   random_seed=1)

    _, _, test = utils.data_process(X_target=test_protein_processed, y=test_target, target_encoding=method,
                                    # drug_encoding= drug_encoding,
                                    split_method='random', frac=[1e-5, 1e-5, 0.99998],
                                    random_seed=1)
    return train, val, test

def load_pair_dataset(dataset_name, path, method):
    # loading pair
    if dataset_name == "PPI_Affinity":
        train = PPI_Affinity(path + '/DeepProtein/data', 'train')
        valid = PPI_Affinity(path + '/DeepProtein/data', 'valid')
        test = PPI_Affinity(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Human_PPI":
        train = HUMAN_PPI(path + '/DeepProtein/data', 'train')
        valid = HUMAN_PPI(path + '/DeepProtein/data', 'valid')
        test = HUMAN_PPI(path + '/DeepProtein/data', 'test')
    elif dataset_name == "Yeast_PPI":
        train = Yeast_PPI(path + '/DeepProtein/data', 'train')
        valid = Yeast_PPI(path + '/DeepProtein/data', 'valid')
        test = Yeast_PPI(path + '/DeepProtein/data', 'test')
    elif dataset_name == "TAP":
        label_list = retrieve_label_name_list('TAP')
        data = Develop(name='TAP', label_name=label_list[0])
        split = data.get_split()
        train_antibody_1, train_antibody_2 = to_two_seq(split, 'train', 'Antibody')
        valid_antibody_1, valid_antibody_2 = to_two_seq(split, 'valid', 'Antibody')
        test_antibody_1, test_antibody_2 = to_two_seq(split, 'test', 'Antibody')
        y_train, y_valid, y_test = split['train']['Y'], split['valid']['Y'], split['test']['Y']
        train = list(zip(train_antibody_1, train_antibody_2, y_train))
        valid = list(zip(valid_antibody_1, valid_antibody_2, y_valid))
        test = list(zip(test_antibody_1, test_antibody_2, y_test))
    elif dataset_name == "SAbDab_Chen":
        data = Develop(name='SAbDab_Chen')
        split = data.get_split()
        train_antibody_1, train_antibody_2 = to_two_seq(split, 'train', 'Antibody', sep=",")
        valid_antibody_1, valid_antibody_2 = to_two_seq(split, 'valid', 'Antibody', sep=",")
        test_antibody_1, test_antibody_2 = to_two_seq(split, 'test', 'Antibody', sep=",")
        y_train, y_valid, y_test = split['train']['Y'], split['valid']['Y'], split['test']['Y']
        train = list(zip(train_antibody_1, train_antibody_2, y_train))
        valid = list(zip(valid_antibody_1, valid_antibody_2, y_valid))
        test = list(zip(test_antibody_1, test_antibody_2, y_test))

    #### deal with targeting (sequence-based and structure-based):
    if method in ['DGL_GAT', 'DGL_GCN', 'DGL_NeuralFP', 'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
        train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train, graph=True, unsqueeze= False)
        valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid, graph=True, unsqueeze= False)
        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test, graph=True, unsqueeze= False)

    else:
        train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train, graph=False, unsqueeze= False)
        valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid, graph=False, unsqueeze= False)
        test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test, graph=False, unsqueeze= False)

    ### deal with pretrained protein language models
    if method == "prot_bert":
        from transformers import BertModel, BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        embedding_model = BertModel.from_pretrained("Rostlab/prot_bert").to("cuda")

    elif method == "esm_1b":
        from transformers import EsmTokenizer, EsmModel
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm1b_t33_650M_UR50S")
        embedding_model = EsmModel.from_pretrained("facebook/esm1b_t33_650M_UR50S").to("cuda")

    elif method == "esm_2":
        from transformers import EsmTokenizer, EsmModel
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        embedding_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to("cuda")

    elif method == "prot_t5":
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
        embedding_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50").to("cuda")
    if method in ["prot_bert", "esm_1b", "esm_2", "prot_t5"]:
        train_protein_1 = get_hf_model_embedding(train_protein_1, tokenizer, embedding_model, method)
        train_protein_2 = get_hf_model_embedding(train_protein_2, tokenizer, embedding_model, method)
        valid_protein_1 = get_hf_model_embedding(valid_protein_1, tokenizer, embedding_model, method)
        valid_protein_2 = get_hf_model_embedding(valid_protein_2, tokenizer, embedding_model, method)
        test_protein_1 = get_hf_model_embedding(test_protein_1, tokenizer, embedding_model, method)
        test_protein_2 = get_hf_model_embedding(test_protein_2, tokenizer, embedding_model, method)

    train, _, _ = data_process(X_target = train_protein_1, X_target_ = train_protein_2, y = train_target,
                    target_encoding = method,
                    split_method='random', frac=[0.99998, 1e-5, 1e-5],
                    random_seed = 1)
    _, val, _ = data_process(X_target = valid_protein_1, X_target_ = valid_protein_2, y = valid_target,
                    target_encoding = method,
                    split_method='random',frac=[1e-5, 0.99998, 1e-5],
                    random_seed = 1)

    _, _, test = data_process(X_target = test_protein_1, X_target_ = test_protein_2, y = test_target,
                    target_encoding = method,
                    split_method='random',frac=[1e-5, 1e-5, 0.99998],
                    random_seed = 1)
    return train, val, test
def load_residue_dataset(dataset_name, path, method):
    # loading residue
    if dataset_name == "PDB":
        data_class, name, X = Epitope, 'PDB_Jespersen', 'Antigen'
        data = data_class(name=name)
        split = data.get_split()
        train_data, valid_data, test_data = split['train'], split['valid'], split['test']
    elif dataset_name == "IEDB":
        data_class, name, X = Epitope, 'IEDB_Jespersen', 'Antigen'
        data = data_class(name=name)
        split = data.get_split()
        train_data, valid_data, test_data = split['train'], split['valid'], split['test']
    elif dataset_name == "SAbDab_Liberis":
        data_class, name, X = Paratope, 'SAbDab_Liberis', 'Antibody'
        data = data_class(name=name)
        split = data.get_split()
        train_data, valid_data, test_data = split['train'], split['valid'], split['test']
    elif dataset_name == "Secondary":
        train_data = SecondaryStructure(path + '/DeepProtein/data', 'train')
        valid_data = SecondaryStructure(path + '/DeepProtein/data', 'valid')
        test_data = SecondaryStructure(path + '/DeepProtein/data', 'cb513')
    ### pre-processing

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

    return train_set, valid_set, test_set



