import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time as T
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import pickle

# torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable
from DeepProtein.LLM_decoders import *

import os

from DeepProtein.utils import *
from DeepProtein.model_helper import Encoder_MultipleLayers, Embeddings
from DeepProtein.encoders import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Classifier(nn.Sequential):
    def __init__(self, model_protein, **config):
        super(Classifier, self).__init__()
        self.input_dim_protein = config['hidden_dim_protein']

        self.model_protein = model_protein

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        if not config['multi']:
            dims = [self.input_dim_protein] + self.hidden_dims + [1]
        else:
            dims = [self.input_dim_protein] + self.hidden_dims + [config['classes']]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_P):
        # each encoding
        v_f = self.model_protein(v_P)
        # concatenate and classify
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f




def model_initialize(**config):
    model = Protein_Prediction(**config)
    return model



def model_pretrained(path_dir=None, model=None):
    if model is not None:
        path_dir = download_pretrained_model(model)
    config = load_dict(path_dir)
    model = Protein_Prediction(**config)
    model.load_pretrained(path_dir + '/model.pt')
    return model


def dgl_collate_func(x):
    x, y = zip(*x)
    import dgl
    x = dgl.batch(x)
    return x, torch.tensor(y)


class Protein_Prediction:
    '''
        Protein Function Prediction
    '''

    def __init__(self, **config):
        target_encoding = config['target_encoding']

        if target_encoding == 'AAC' or target_encoding == 'PseudoAAC' or target_encoding == 'Conjoint_triad' or target_encoding == 'Quasi-seq' or target_encoding == 'ESPF':
            self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'],
                                     config['mlp_hidden_dims_target'])
        elif target_encoding == 'CNN':
            self.model_protein = CNN('protein', **config)
        elif target_encoding == 'CNN_RNN':
            self.model_protein = CNN_RNN('protein', **config)
        elif target_encoding == 'Transformer':
            self.model_protein = transformer('protein', **config)
        elif target_encoding == 'DGL_GCN':
            self.model_protein = DGL_GCN(in_feats=74,
                                         hidden_feats=[config['gnn_hid_dim_drug']] * config['gnn_num_layers'],
                                         activation=[config['gnn_activation']] * config['gnn_num_layers'],
                                         predictor_dim=config['hidden_dim_drug'])
        elif target_encoding == 'DGL_GAT':
            self.model_protein = DGL_GAT(in_feats=74,
                                         hidden_feats=[config['gnn_hid_dim_drug']] * config['gnn_num_layers'],
                                         activation=[config['gnn_activation']] * config['gnn_num_layers'],
                                         predictor_dim=config['hidden_dim_drug'])
        elif target_encoding == 'DGL_NeuralFP':
            self.model_protein = DGL_NeuralFP(in_feats = 74,
									hidden_feats = [config['gnn_hid_dim_drug']] * config['gnn_num_layers'],
									max_degree = config['neuralfp_max_degree'],
									activation = [config['gnn_activation']] * config['gnn_num_layers'],
									predictor_hidden_size = config['neuralfp_predictor_hid_dim'],
									predictor_dim = config['hidden_dim_drug'],
									predictor_activation = config['neuralfp_predictor_activation'])
        elif target_encoding == 'DGL_AttentiveFP':
            self.model_protein = DGL_AttentiveFP(node_feat_size=39,
                                                edge_feat_size=11,
                                                 num_layers=config['gnn_num_layers'],
                                                 num_timesteps=config['attentivefp_num_timesteps'],
                                                 graph_feat_size=config['gnn_hid_dim_drug'],
                                                 predictor_dim=config['hidden_dim_drug'],
                                             )
        elif target_encoding == 'DGL_MPNN':
            self.model_protein = DGL_MPNN(node_feat_size = 74,
                                    edge_feat_size = 13,
                                    num_timesteps = 1,
                                    graph_feat_size = config['gnn_hid_dim_drug'],
                                    predictor_dim = config['hidden_dim_drug']
        )
        elif target_encoding == 'PAGTN':
            self.model_protein = PAGTN(node_feat_size = 74,
                                       node_hid_size = config['gnn_hid_dim_drug'],
                                       edge_feat_size = 13,
                                       graph_feat_size = config['gnn_hid_dim_drug'],
                                       predictor_dim=config['hidden_dim_drug'])

        elif target_encoding == 'Graphormer':
            self.model_protein = Graphormer(node_feat_size=74,
                                            node_hid_size=config['gnn_hid_dim_drug'],
                                             graph_feat_size=config['gnn_hid_dim_drug'],
                                             predictor_dim=config['hidden_dim_drug'])
        elif target_encoding == 'prot_bert':
            self.model_protein = Prot_Bert_Predictor('protein', **config)

        elif target_encoding == 'esm_1b':
            self.model_protein = ESM_1B_Predictor('protein', **config)

        elif target_encoding == 'esm_2':
            self.model_protein = ESM_2_Predictor('protein', **config)

        elif target_encoding == 'prot_t5':
            self.model_protein = Prot_T5_Predictor('protein', **config)



        elif target_encoding in ['BioMistral', 'BioT5_plus', 'ChemLLM_7B', 'LlaSMol', 'ChemDFM']:
            self.model_protein = Prot_Bert_Predictor('protein', **config)

        else:
            raise AttributeError('Please use one of the available encoding method.')

        self.model = Classifier(self.model_protein, **config)
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.target_encoding = target_encoding
        self.result_folder = config['result_folder']
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.binary = config['binary']
        self.multi = config['multi']
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0


    def test_LLM(self, data, y_label, dataset_name, repurposing_mode=False):
        model = None
        if self.target_encoding == 'BioMistral':
            model = BioMistral(dataset_name)
        elif self.target_encoding == 'BioT5_plus':
            model = BioT5_plus(dataset_name)
        elif self.target_encoding == "ChemLLM_7B":
            model = ChemLLM_7B(dataset_name)
        elif self.target_encoding == "LlaSMol":
            model = LlaSMol(dataset_name)
        elif self.target_encoding == "ChemDFM":
            model = ChemDFM(dataset_name)

        y_pred = model.inference(data)
        if self.binary:
            if repurposing_mode:
                return y_pred

            return roc_auc_score(y_label, y_pred, average='macro'), average_precision_score(y_label, y_pred, average='macro'), f1_score(y_label,
                                                                                                      y_pred, average='macro'), y_pred
        elif self.multi:
            if repurposing_mode:
                return y_pred

            return accuracy_score(y_label, y_pred), 0, f1_score(y_label,y_pred, average='macro'), y_pred

        else:
            y_label = y_label.squeeze()
            y_pred = y_pred.squeeze()
            if repurposing_mode:
                return y_pred
            elif self.config['use_spearmanr']:
                return mean_absolute_error(y_label, y_pred), mean_squared_error(y_label, y_pred), \
                    spearmanr(y_label, y_pred)[0], \
                    spearmanr(y_label, y_pred)[1], \
                    concordance_index(y_label, y_pred), y_pred
            return mean_absolute_error(y_label, y_pred), mean_squared_error(y_label, y_pred), \
                pearsonr(y_label, y_pred)[0], \
                pearsonr(y_label, y_pred)[1], \
                concordance_index(y_label, y_pred), y_pred


    def LLM_test_and_log(self, data, y_label, dataset_name, repurposing_mode, verbose=True):
        if self.binary:
            auc, auprc, f1, logits = self.test_LLM(data, y_label, dataset_name)
            test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
            float2str = lambda x: '%0.4f' % x
            test_table.add_row(list(map(float2str, [auc, auprc, f1])))
            wandb.log({"TEST AUROC": auc, "TEST AUPRC": auprc, "TEST F1": f1})
            if verbose:
                print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))
        elif self.multi:
            acc, auprc, f1, logits = self.test_LLM(data, y_label, dataset_name)
            test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
            float2str = lambda x: '%0.4f' % x
            test_table.add_row(list(map(float2str, [acc, auprc, f1])))
            wandb.log({"TEST Accuracy": acc, "TEST AUPRC": auprc, "TEST F1": f1})
            if verbose:
                print('Testing Accuracy: ' + str(acc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))
        else:
            mae, mse, r2, p_val, CI, logits = self.test_LLM(data, y_label, dataset_name)
            test_table = PrettyTable(["MAE", "MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
            float2str = lambda x: '%0.4f' % x
            test_table.add_row(list(map(float2str, [mae, mse, r2, p_val, CI])))
            wandb.log({"TEST MSE": mse, "MAE": mae, "TEST R2": r2, "TEST p_val": p_val, "TEST Concordance Index": CI})
            if verbose:
                if self.config['use_spearmanr']:
                    print('Testing MSE: ' + str(mse) + ' , MAE: ' + str(mae) + ' , Spearman Correlation: ' + str(r2)
                          + ' with p-value: ' + str(f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI))
                else:
                    print('Testing MSE: ' + str(mse) + ' , MAE: ' + str(mae) + ' , Pearson Correlation: ' + str(r2)
                          + ' with p-value: ' + str(f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI))


        prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
        with open(prettytable_file, 'w') as fp:
            fp.write(test_table.get_string())


    def test_(self, data_generator, model, repurposing_mode=False, test=False, verbose=True):
        y_pred = []
        multi_y_pred = []
        y_label = []
        model.eval()
        for i, (v_p, label) in enumerate(data_generator):
            if self.target_encoding in ['Transformer', 'DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP',
                                        'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
                v_p = v_p
            else:
                v_p = v_p.float().to(self.device)
            score = self.model(v_p)

            if self.binary:
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score)).detach().cpu().numpy()
            elif self.multi:
                m = torch.nn.Softmax(dim=-1)
                logits = torch.squeeze(m(score)).detach().cpu().numpy()
            else:
                logits = torch.squeeze(score).detach().cpu().numpy()

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()


            if self.multi:
                multi_y_pred = multi_y_pred + logits.tolist()
                multi_outputs = np.argmax(np.asarray(multi_y_pred), axis=-1)
            else:
                y_pred = y_pred + logits.flatten().tolist()
                outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

        if self.multi:
            y_label = np.array(y_label).astype(int)
            # multi_y_pred = np.array(multi_y_pred).astype(int)


        # print(accuracy_score(y_label, multi_outputs))


        model.train()
        if self.binary:
            if repurposing_mode:
                return y_pred
            ## ROC-AUC curve
            if test:
                if verbose:
                    roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
                    plt.figure(0)
                    roc_curve(y_pred, y_label, roc_auc_file, self.target_encoding)
                    plt.figure(1)
                    pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
                    prauc_curve(y_pred, y_label, pr_auc_file, self.target_encoding)

            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                      outputs), y_pred
        elif self.multi:
            if repurposing_mode:
                return multi_outputs

            if test:
                if verbose:
                    confusion_matrix_file = os.path.join(self.result_folder, "confusion_matrix.jpg")
                    plt.figure(0)
                    plot_confusion_matrix(multi_outputs, y_label, confusion_matrix_file, self.target_encoding)

            multi_outputs = multi_outputs
            return accuracy_score(y_label, multi_outputs), 0, f1_score(y_label,
                                                                                multi_outputs, average='macro'), multi_y_pred


        else:
            if repurposing_mode:
                return y_pred
            elif self.config['use_spearmanr']:
                return mean_absolute_error(y_label, y_pred), mean_squared_error(y_label, y_pred), \
                    spearmanr(y_label, y_pred)[0], \
                    spearmanr(y_label, y_pred)[1], \
                    concordance_index(y_label, y_pred), y_pred
            return  mean_absolute_error(y_label, y_pred), mean_squared_error(y_label, y_pred), \
                pearsonr(y_label, y_pred)[0], \
                pearsonr(y_label, y_pred)[1], \
                concordance_index(y_label, y_pred), y_pred

    def train_LLM(self, test):
        ### Fine-tune LLM
        ##TODO:
        pass


    def train(self, train, val, test=None, verbose=True, compute_pos_enc=False):

        if len(train.Label.unique()) == 2:
            self.binary = True
            self.config['binary'] = True

        lr = self.config['LR']
        decay = self.config['decay']

        BATCH_SIZE = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        if 'test_every_X_epoch' in self.config.keys():
            test_every_X_epoch = self.config['test_every_X_epoch']
        else:
            test_every_X_epoch = 40
        loss_history = []

        self.model = self.model.to(self.device)

        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = nn.DataParallel(self.model, dim=0)
        elif torch.cuda.device_count() == 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            if verbose:
                print("Let's use CPU/s!")
        # Future TODO: support multiple optimizers with parameters
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)

        if verbose:
            print('--- Data Preparation ---')

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False}

        if self.target_encoding in ['DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP', 'DGL_AttentiveFP',
                                    'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
            params['collate_fn'] = dgl_collate_func

        training_generator = data.DataLoader(data_process_loader_Protein_Prediction(train.index.values,
                                                                                    train.Label.values,
                                                                                    train, **self.config),
                                             **params)





        validation_generator = data.DataLoader(data_process_loader_Protein_Prediction(val.index.values,
                                                                                      val.Label.values,
                                                                                      val, **self.config),
                                               **params)



        if test is not None:

            info = data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config)
            params_test = {'batch_size': BATCH_SIZE,
                           'shuffle': False,
                           'num_workers': self.config['num_workers'],
                           'drop_last': False,
                           'sampler': SequentialSampler(info)}

            if self.target_encoding in ['DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP', 'DGL_AttentiveFP',
                                        'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
                params_test['collate_fn'] = dgl_collate_func

            testing_generator = data.DataLoader(
                data_process_loader_Protein_Prediction(test.index.values, test.Label.values, test, **self.config),
                **params_test)


        if compute_pos_enc:
            print("========= Computing Positional Encoding ..... =========")
            training_generator = compute_pos(training_generator, **params)
            validation_generator = compute_pos(validation_generator, **params)
            testing_generator = compute_pos(testing_generator, **params)


        # early stopping
        if self.binary:
            max_auc = 0
        elif self.multi:
            max_acc = 0
        else:
            max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"]
        if self.binary:
            valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
        elif self.multi:
            valid_metric_header.extend(["ACC", "AUPRC", "F1"])
        else:
            valid_metric_header.extend(["MAE", "MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x

        if verbose:
            print('--- Go for Training ---')
        t_start = T()

        self.model.train()
        for epo in range(train_epoch):

            for i, (v_p, label) in enumerate(training_generator):
                if self.target_encoding in ['Transformer', 'DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP',
                                            'DGL_AttentiveFP', 'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
                    v_p = v_p
                else:
                    # print(v_p)
                    v_p = v_p.float().to(self.device)



                score = self.model(v_p)
                label = torch.from_numpy(np.array(label)).float().to(self.device)
                if self.binary:
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(score), 1)
                    # label = torch.squeeze(label, 1)
                    if label.dim() > 1:
                        label = torch.squeeze(label)
                    loss = loss_fct(n, label)

                elif self.multi:

                    loss_fct = torch.nn.NLLLoss()
                    m = torch.nn.LogSoftmax(dim=-1)

                    n = m(score)
                    label = torch.squeeze(label).long()  # This will remove all dimensions with size 1

                    # Check if the tensor still has a second dimension before trying to squeeze it
                    if label.dim() > 1:
                        label = label.squeeze(1)
                    loss = loss_fct(n, label)

                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(score, 1)
                    if self.target_encoding not in ['DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP', 'DGL_AttentiveFP',
                                                    'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
                        label = torch.squeeze(label, 1)
                    loss = loss_fct(n, label)

                loss_history.append(loss.item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 100 == 0):
                        t_now = T()
                        if verbose:
                            print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                                  ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                                  ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")
                        ### record total run time
            wandb.log({"training loss": loss.cpu().detach().numpy()})
            ##### validate, select the best model up to now
            with torch.set_grad_enabled(False):
                if self.binary:
                    ## binary: ROC-AUC, PR-AUC, F1
                    auc, auprc, f1, logits = self.test_(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str, [auc, auprc, f1]))
                    valid_metric_record.append(lst)
                    if auc > max_auc:
                        model_max = copy.deepcopy(self.model)
                        max_auc = auc
                    wandb.log({"epoch": epo + 1, "AUROC": auc, "AUPRC": auprc, "F1": f1})
                    if verbose:
                        print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                              ' , AUPRC: ' + str(auprc)[:7] + ' , F1: ' + str(f1)[:7])

                elif self.multi:

                    acc, auprc, f1, logits = self.test_(validation_generator, self.model)
                    lst = ["epoch " + str(epo)] + list(map(float2str, [acc, auprc, f1]))
                    valid_metric_record.append(lst)
                    if acc > max_acc:
                        model_max = copy.deepcopy(self.model)
                        max_acc = acc
                    wandb.log({"epoch": epo + 1, "Accuracy": acc, "AUPRC": auprc, "F1": f1})
                    if verbose:
                        print('Validation at Epoch ' + str(epo + 1) + ' , Accuracy: ' + str(acc)[:7] + \
                              ' , AUPRC: ' + str(auprc)[:7] + ' , F1: ' + str(f1)[:7])

                else:
                    ### regression: MSE, Pearson Correlation, with p-value, Concordance Index

                    mae, mse, r2, p_val, CI, logits = self.test_(validation_generator, self.model)

                    lst = ["epoch " + str(epo)] + list(map(float2str, [mae, mse, r2, p_val, CI]))
                    valid_metric_record.append(lst)
                    if mse < max_MSE:
                        model_max = copy.deepcopy(self.model)
                        max_MSE = mse
                    wandb.log({"epoch": epo + 1, "MAE": mae, "MSE": mse, "R2": r2, "p_val": p_val, "Concordance Index": CI})
                    if verbose:
                        if self.config['use_spearmanr']:
                            print('Validation at Epoch ' + str(epo + 1) + ' , MAE: ' + str(mae)[:7] + ' , MSE: ' + str(mse)[:7] + ' , Spearman Correlation: ' \
                                  + str(r2)[:7] + ' with p-value: ' + str(
                                f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI)[:7])
                        else:
                            print('Validation at Epoch ' + str(epo + 1) + ' , MAE: ' + str(mae)[:7] + ' , MSE: ' + str(mse)[:7] + ' , Pearson Correlation: ' \
                                  + str(r2)[:7] + ' with p-value: ' + str(
                                f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI)[:7])
            table.add_row(lst)

        #### after training
        prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
        with open(prettytable_file, 'w') as fp:
            fp.write(table.get_string())

        # load early stopped model
        self.model = model_max

        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
            if self.binary:
                auc, auprc, f1, logits = self.test_(testing_generator, model_max, test=True, verbose=verbose)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [auc, auprc, f1])))
                wandb.log({"TEST AUROC": auc, "TEST AUPRC": auprc, "TEST F1": f1})
                if verbose:
                    print('Testing AUROC: ' + str(auc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))

            elif self.multi:
                acc, auprc, f1, logits = self.test_(testing_generator, model_max, test=True, verbose=verbose)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [acc, auprc, f1])))
                wandb.log({"TEST Accuracy": acc, "TEST AUPRC": auprc, "TEST F1": f1})
                if verbose:
                    print('Testing Accuracy: ' + str(acc) + ' , AUPRC: ' + str(auprc) + ' , F1: ' + str(f1))

            else:
                mae, mse, r2, p_val, CI, logits = self.test_(testing_generator, model_max, test=True, verbose=verbose)
                test_table = PrettyTable(["MAE", "MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
                test_table.add_row(list(map(float2str, [mae, mse, r2, p_val, CI])))
                wandb.log({"TEST MSE": mse, "MAE": mae,  "TEST R2": r2, "TEST p_val": p_val, "TEST Concordance Index": CI})
                if verbose:
                    if self.config['use_spearmanr']:
                        print('Testing MSE: ' + str(mse) + ' , MAE: ' + str(mae) + ' , Spearman Correlation: ' + str(r2)
                              + ' with p-value: ' + str(f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI))
                    else:
                        print('Testing MSE: ' + str(mse) + ' , MAE: ' + str(mae) + ' , Pearson Correlation: ' + str(r2)
                              + ' with p-value: ' + str(f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI))
            np.save(os.path.join(self.result_folder, str(self.target_encoding)
                                 + '_logits.npy'), np.array(logits))

            ######### learning record ###########

            ### 1. test results
            prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        if verbose:
            ### 2. learning curve
            fontsize = 16
            iter_num = list(range(1, len(loss_history) + 1))
            plt.figure(3)
            plt.plot(iter_num, loss_history, "bo-")
            plt.xlabel("iteration", fontsize=fontsize)
            plt.ylabel("loss value", fontsize=fontsize)
            pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
            with open(pkl_file, 'wb') as pck:
                pickle.dump(loss_history, pck)

            fig_file = os.path.join(self.result_folder, "loss_curve.png")
            plt.savefig(fig_file)
        if verbose:
            print('--- Training Finished ---')

    def predict(self, df_data, verbose=True):
        '''
            utils.data_process_repurpose_virtual_screening
            pd.DataFrame
        '''
        if verbose:
            print('predicting...')
        info = data_process_loader_Protein_Prediction(df_data.index.values, df_data.Label.values, df_data,
                                                      **self.config)
        self.model.to(device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}

        if self.target_encoding in ['DGL_GCN', 'DGL_GAT', 'DGL_NeuralFP', 'DGL_AttentiveFP',
                                    'DGL_MPNN', 'PAGTN', 'EGT', 'Graphormer']:
            params['collate_fn'] = dgl_collate_func

        generator = data.DataLoader(info, **params)

        score = self.test_(generator, self.model, repurposing_mode=True)
        # set repurposing mode to true, will return only the scores.
        return score

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        if self.device == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        # to support training from multi-gpus data-parallel:

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config['binary']
        self.multi = self.config['multi']