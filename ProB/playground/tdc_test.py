import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
from copy import deepcopy
torch.manual_seed(1)
from tdc.single_pred import Epitope, Paratope

# data_class, name, X = Epitope, 'IEDB_Jespersen', 'Antigen'
# data_class, name, X = Epitope, 'PDB_Jespersen', 'Antigen'
data_class, name, X = Paratope, 'SAbDab_Liberis', 'Antibody'

data = data_class(name = name)
split = data.get_split()
train_data = split['train']
valid_data = split['valid']
test_data = split['test']
vocab_set = set()


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score


def plot(label_lst, predict_lst, name):

    fpr, tpr, thresholds = metrics.roc_curve(label_lst, predict_lst, )
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(name)



############################# vocab #############################
def data2vocab(data, train_data):
    vocab_set = set()
    total_length, positive_num = 0, 0
    for i in range(len(data)):
        antigen = train_data[X][i]
        vocab_set = vocab_set.union(set(antigen))
        Y = train_data['Y'][i]
        assert len(antigen) > max(Y)
        total_length += len(antigen)
        positive_num += len(Y)
    return vocab_set, positive_num / total_length

train_vocab, train_positive_ratio = data2vocab(train_data)
valid_vocab, valid_positive_ratio = data2vocab(valid_data)
test_vocab, test_positive_ratio = data2vocab(test_data)
print(train_positive_ratio)
vocab_set = train_vocab.union(valid_vocab)
vocab_set = vocab_set.union(test_vocab)
vocab_lst = list(vocab_set)
############################# vocab #############################
def onehot(idx, length):
    lst = [0 for i in range(length)]
    lst[idx] = 1
    return lst

def zerohot(length):
    return [0 for i in range(length)]

def standardize_data(data, vocab_lst, maxlength = 300):
    length = len(data)
    standard_data = []
    for i in range(length):
        antigen = data[X][i]
        Y = data['Y'][i]
        sequence = [onehot(vocab_lst.index(s), len(vocab_lst)) for s in antigen]
        labels = [0 for i in range(len(antigen))]
        mask = [True for i in range(len(labels))]
        sequence += (maxlength-len(sequence)) * [zerohot(len(vocab_lst))]
        labels += (maxlength-len(labels)) * [0]
        mask += (maxlength-len(mask)) * [False]
        for y in Y:
            labels[y] = 1
        sequence, labels, mask = sequence[:maxlength], labels[:maxlength], mask[:maxlength]
        sequence, labels, mask = torch.FloatTensor(sequence), torch.FloatTensor(labels), torch.BoolTensor(mask)
        # print(sequence.shape, labels.shape, mask.shape)
        standard_data.append((sequence, labels, mask))
    return standard_data

train_data = standardize_data(train_data, vocab_lst)
valid_data = standardize_data(valid_data, vocab_lst)
test_data = standardize_data(test_data, vocab_lst)


class dataset(Dataset):
    def __init__(self, data):
        self.sequences = [i[0] for i in data]
        self.labels = [i[1] for i in data]
        self.mask = [i[2] for i in data]

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index], self.mask[index]

    def __len__(self):
        return len(self.labels)

train_set = dataset(train_data)
valid_set = dataset(valid_data)
test_set = dataset(test_data)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

#################################################################
############################# model #############################
#################################################################

class RNN(nn.Module):
    def __init__(self, name, hidden_size, input_size, num_layers = 2):
        super(RNN, self).__init__()
        self.name = name
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=hidden_size,         # rnn hidden unit
            num_layers=num_layers,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(hidden_size, 1)
        criterion = torch.nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        print(r_out.shape) # 16 * 300 * 100

        # choose r_out at the last time step
        out = self.out(r_out)
        out = out.squeeze(-1) # 16 * 300
        # print(out.shape)
        return out

    def learn(self, sequence, labels, mask):
        prediction = self.forward(sequence)
        # print("size", prediction.shape, labels.shape, mask.shape)
        criterion = torch.nn.BCEWithLogitsLoss(size_average=True, weight = mask)
        loss = criterion(prediction, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self, test_loader, name):
        label_lst, prediction_lst = [], []
        for sequence, labels, mask in test_loader:
            prediction = self.forward(sequence)
            prediction = torch.sigmoid(prediction)
            for pred, label, msk in zip(prediction, labels, mask):
                num = sum(msk.tolist())
                pred = pred.tolist()[:num]
                label = label.tolist()[:num]
                label_lst.extend(label)
                prediction_lst.extend(pred)
        sort_pred = deepcopy(prediction_lst)
        sort_pred.sort()
        threshold = sort_pred[int(len(sort_pred)*0.9)]
        float2binary = lambda x:0 if x<threshold else 1
        binary_pred_lst = list(map(float2binary, prediction_lst))
        plot(label_lst, prediction_lst, name)
        print('roc_auc', roc_auc_score(label_lst, prediction_lst),
              'F1', f1_score(label_lst, binary_pred_lst),
              'prauc', average_precision_score(label_lst, binary_pred_lst))


class CNN(nn.Module):
    def __init__(self, name, input_channels, num_filters, kernel_size, hidden_dim=128, output_size=1):
        super(CNN, self).__init__()
        self.name = name

        # CNN layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=num_filters, kernel_size=kernel_size,
                               padding='same')

        # Pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, padding=0)

        # Fully connected layer
        self.fc1 = nn.Linear(num_filters, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

        # Optimizer and loss function
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # x should have shape (batch_size, sequence_length, input_channels)
        # We need to transpose it to (batch_size, input_channels, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)  # Reshape x to (batch_size, input_channels, sequence_length)

        # Pass through the convolutional layer
        x = self.conv1(x)
        x = F.relu(x)

        # Pass through the pooling layer
        x = self.pool(x)  # x now has shape (batch_size, num_filters, sequence_length // 2)

        # Transpose back for compatibility with the fully connected layers
        x = x.permute(0, 2, 1)  # Reshape to (batch_size, sequence_length // 2, num_filters)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Squeeze the output channel if needed (batch_size, sequence_length // 2, output_size)
        x = x.squeeze(-1)

        return x

    def learn(self, sequence, labels, mask):
        prediction = self.forward(sequence)

        # Adjust mask and labels to match the output shape
        mask = mask[:, :prediction.shape[1]]
        labels = labels[:, :prediction.shape[1]]

        loss = self.criterion(prediction, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def test(self, test_loader, name):
        label_lst, prediction_lst = [], []
        for sequence, labels, mask in test_loader:
            prediction = self.forward(sequence)
            prediction = torch.sigmoid(prediction)
            for pred, label, msk in zip(prediction, labels, mask):
                num = sum(msk.tolist())
                pred = pred.tolist()[:num]
                label = label.tolist()[:num]
                label_lst.extend(label)
                prediction_lst.extend(pred)

        sort_pred = deepcopy(prediction_lst)
        sort_pred.sort()
        threshold = sort_pred[int(len(sort_pred) * 0.9)]
        float2binary = lambda x: 0 if x < threshold else 1
        binary_pred_lst = list(map(float2binary, prediction_lst))
        plot(label_lst, prediction_lst, name)
        print('roc_auc', roc_auc_score(label_lst, prediction_lst),
              'F1', f1_score(label_lst, binary_pred_lst),
              'prauc', average_precision_score(label_lst, binary_pred_lst))

#################################################################
############################# learn #############################
#################################################################

model = RNN(name = 'Epitope', hidden_size=100, input_size = len(vocab_lst))

# input_channels = len(vocab_lst)  # Number of unique characters in your vocabulary
# model = CNN(name='Epitope', input_channels=input_channels, num_filters=32, kernel_size=5, hidden_dim=128)
epoch = 10
for ep in range(epoch):
    for sequence, labels, mask in train_loader:
        model.learn(sequence, labels, mask)
    model.test(test_loader, name = model.name + "_" + str(ep) + ".png")