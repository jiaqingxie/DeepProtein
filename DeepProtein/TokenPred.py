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
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_absolute_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import pickle

# torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

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
        # print(v_P.shape) # batch_size * max_length * 24
        v_f = self.model_protein(v_P)

        # concatenate and classify
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        # print(v_f.shape)
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


# def dgl_collate_func(x):
#     x, y = zip(*x)
#     import dgl
#     x = dgl.batch(x)
#     return x, torch.tensor(y)


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
        elif target_encoding == 'Token_CNN':
            self.model_protein = Token_CNN('protein', **config)
        elif target_encoding == 'CNN_RNN':
            self.model_protein = CNN_RNN('protein', **config)
        elif target_encoding == 'Token_CNN_RNN':
            self.model_protein = Token_CNN_RNN('protein', **config)
        elif target_encoding == 'Transformer':
            self.model_protein = transformer('protein', **config)
        elif target_encoding == 'Token_Transformer':
            self.model_protein = Token_Transformer('protein', **config)
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

    def test_(self, data_generator, model, repurposing_mode=False, test=False, verbose=True):
        """
        Evaluate the model on a given data generator.

        :param data_generator: An iterable that yields (v_p, labels, mask) per batch.
        :param model: The model to be evaluated.
        :param repurposing_mode: If True, returns raw predictions (for e.g. drug repurposing).
        :param test: If True, can trigger additional plotting or saving of results.
        :param verbose: If True, prints or plots additional logs.

        :return: For binary:
                   (roc_auc, pr_auc, f1, prediction_list)
                 For multi-class:
                   (accuracy, 0, f1_macro, y_pred_list)
        """
        model.eval()

        # For binary classification
        label_lst, prediction_lst = [], []

        # For multi-class classification
        multi_outputs = []  # Will store raw probability distributions
        multi_y_pred = []  # Will store the discrete predictions (argmax)
        y_label = []  # Will store the true labels

        for i, (v_p, labels, mask) in enumerate(data_generator):
            # Move input features to the GPU if needed
            if self.target_encoding in ['Transformer']:
                # Possibly skip float conversion if already in correct dtype
                v_p = v_p
            else:
                v_p = v_p.float().to(self.device)

            # Get the model outputs (logits)
            prediction = model(v_p)

            # ---------------------- Binary Classification ---------------------- #
            if self.binary:
                # Apply sigmoid for binary classification
                prediction = torch.sigmoid(prediction)

                # Collect the predictions and labels using the mask
                for pred, label, msk in zip(prediction, labels, mask):
                    num = sum(msk.tolist())  # how many valid entries in this sequence
                    pred = pred.tolist()[:num]
                    label = label.tolist()[:num]
                    label_lst.extend(label)
                    prediction_lst.extend(pred)

            # --------------------- Multi-Class Classification --------------------- #
            elif self.multi:
                # Apply softmax for multi-class classification
                prediction = F.softmax(prediction, dim=-1)

                # Collect the predictions and labels using the mask
                for pred, label, msk in zip(prediction, labels, mask):
                    num = sum(msk.tolist())  # how many valid entries in this sequence

                    # pred is shape [num_classes] if it's a single classification
                    # or [seq_len, num_classes] if there's a sequence dimension.
                    # We'll truncate by 'num' the same way as in binary:
                    truncated_pred = pred[:num]
                    truncated_label = label[:num]

                    # Get discrete predictions
                    # If truncated_pred is shape [num, num_classes], do argmax over dim=-1
                    # If it's a single item shape [num_classes], then `truncated_pred.unsqueeze(0)` may be needed.
                    # For consistency, assume a [seq_len, num_classes] shape:
                    predicted_classes = truncated_pred.argmax(dim=-1)

                    # Store raw probabilities (flattened) and predicted classes
                    multi_outputs.extend(truncated_pred.cpu().numpy().tolist())
                    multi_y_pred.extend(predicted_classes.cpu().numpy().tolist())
                    y_label.extend(truncated_label.cpu().numpy().tolist())

        # Switch back to train mode after evaluation
        model.train()

        # ------------------------- Return for Binary ------------------------- #
        if self.binary:
            if repurposing_mode:
                # Return continuous predictions (useful for further processing)
                return prediction_lst

            # Determine the threshold for binary classification (90th percentile)
            sort_pred = copy.deepcopy(prediction_lst)
            sort_pred.sort()
            threshold = sort_pred[int(len(sort_pred) * 0.9)]
            float2binary = lambda x: 0 if x < threshold else 1
            binary_pred_lst = list(map(float2binary, prediction_lst))

            # If test mode, optionally plot or save AUC curves
            if test and verbose:
                roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
                plt.figure(0)
                roc_curve(label_lst, binary_pred_lst, roc_auc_file, self.target_encoding)
                plt.figure(1)
                pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
                prauc_curve(label_lst, binary_pred_lst, pr_auc_file, self.target_encoding)

            # Compute binary metrics
            return (
                roc_auc_score(label_lst, binary_pred_lst),
                average_precision_score(label_lst, binary_pred_lst),
                f1_score(label_lst, binary_pred_lst),
                prediction_lst
            )

        # ------------------------ Return for Multi-Class --------------------- #
        elif self.multi:
            if repurposing_mode:
                # Return raw probability distributions (if needed for repurposing)
                return multi_outputs

            # If test mode, optionally plot a confusion matrix
            if test and verbose:
                confusion_matrix_file = os.path.join(self.result_folder, "confusion_matrix.jpg")
                plt.figure(0)
                # Usually you'd pass discrete predictions vs. true labels:
                plot_confusion_matrix(multi_y_pred, y_label, confusion_matrix_file, self.target_encoding)

            # Compute multi-class metrics
            acc = accuracy_score(y_label, multi_y_pred)
            f1_macro = f1_score(y_label, multi_y_pred, average='macro')

            # Return (accuracy, 0, macro-F1, predicted_labels)
            return acc, 0, f1_macro, multi_y_pred
        else:
            raise NotImplementedError("Not implemented yet")

    def train(self, train, val, test=None, verbose=True, batch_size=32):

        # if len(train.Label.unique()) == 2:
        #     self.binary = True
        #     self.config['binary'] = True

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


        training_generator = data.DataLoader(data_process_loader_Token_Protein_Prediction(train), batch_size = batch_size)

        validation_generator = data.DataLoader(data_process_loader_Token_Protein_Prediction(val), batch_size = batch_size)

        if test is not None:

            info = data_process_loader_Token_Protein_Prediction(test)
            params_test = {'batch_size': BATCH_SIZE,
                           'shuffle': False,
                           'num_workers': self.config['num_workers'],
                           'drop_last': False,
                           'sampler': SequentialSampler(info)}

            testing_generator = data.DataLoader(
                data_process_loader_Token_Protein_Prediction(test), batch_size=batch_size)



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
            valid_metric_header.extend(["Acc", "AUPRC", "F1"])
        else:
            valid_metric_header.extend(["MAE", "MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x

        if verbose:
            print('--- Go for Training ---')
        t_start = time()

        self.model.train()
        for epo in range(train_epoch):
            self.model.train()  # Ensure the model is in training mode

            for i, (sequence, label, mask) in enumerate(training_generator):
                if self.target_encoding in ['Transformer']:
                    sequence = sequence
                else:
                    sequence = sequence.float().to(self.device)
                    mask = mask.to(self.device)

                score = self.model(sequence)

                # print(score.shape)

                if not torch.is_tensor(label):
                    label = torch.from_numpy(np.array(label)).float().to(self.device)
                else:
                    label = label.float().to(self.device)

                if self.binary:

                    if score.dim() > 2:
                        score = score.squeeze(-1)

                    if label.dim() > 1 and label.size(-1) == 1:
                        label = label.squeeze(-1)

                    criterion = torch.nn.BCEWithLogitsLoss(weight=mask, reduction='mean')

                    # print(label.shape)
                    loss = criterion(score, label)
                elif self.multi:

                    if score.dim() > 2:
                        score = score

                    if label.dim() > 1 and label.size(-1) == 1:

                        label = label.squeeze(-1)

                    criterion = torch.nn.CrossEntropyLoss()

                    # probs = torch.softmax(score, dim=-1)
                    # predicted_labels = torch.argmax(probs, dim=-1)

                    loss = criterion(score.view(-1, 3), label.view(-1).long())

                else:
                    raise NotImplementedError("Not Implemented for non-binary settings")

                # Store the loss value
                loss_history.append(loss.item())

                # Backward pass and optimization
                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 100 == 0):
                        t_now = time()
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
                   raise NotImplementedError("Not Implemented Yet")
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
                raise NotImplementedError("Not Implemented Yet")

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
        info = data_process_loader_Token_Protein_Prediction(df_data)
        self.model.to(device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}

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