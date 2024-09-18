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

from DeepPurpose import PPI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

X_targets, X_targets_, y = read_file_training_dataset_protein_protein_pairs("toy_data/ppi.txt")

target_encoding = 'CNN'
train, val, test = data_process(X_target = X_targets, X_target_ = X_targets_, y = y,
			    target_encoding = target_encoding,
			    split_method='random',
			    random_seed = 1)

config = generate_config(target_encoding = target_encoding,
                         cls_hidden_dims = [512],
                         train_epoch = 20,
                         LR = 0.001,
                         batch_size = 128,
                        )

model = models.model_initialize(**config)
model.train(train, val, test)