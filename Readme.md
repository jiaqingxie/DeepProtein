<p align="center"><img src="figs/deeppurpose_pp_logo.png" alt="logo" width="400px" /></p>

<h3 align="center">
<p> DeepProtein: Deep Learning Library and Benchmark for Protein Sequence Learning <br></h3>
<h4 align="center">
<p>  Applications in Protein Property Prediction, Localization Prediction, Protein-Protein Interaction, antigen epitope prediction, antibody paratope prediction, antibody developability prediction, etc. </h4>

---

[![PyPI version](https://badge.fury.io/py/DeepProtein.svg)](https://pypi.org/project/DeepProtein/)
[![Downloads](https://pepy.tech/badge/DeepProtein/month)](https://pepy.tech/project/DeepProtein)
[![Downloads](https://pepy.tech/badge/DeepProtein)](https://pepy.tech/project/DeepProtein)
[![GitHub Repo stars](https://img.shields.io/github/stars/jiaqingxie/DeepPurposePlusPlus)](https://github.com/jiaqingxie/DeepPurposePlusPlus/stargazers)
[![GitHub Repo forks](https://img.shields.io/github/forks/jiaqingxie/DeepPurposePlusPlus)](https://github.com/jiaqingxie/DeepPurposePlusPlus/network/members)

## News
- [09/24] DeepProtein is online. The website is under initial test and construction. 


## Installation

First, we recommend you follow the instructions on how DeepPurpose's dependencies are installed.
```bash
conda create -n DeepProtein python=3.9
conda activate DeepProtein
pip install git+https://github.com/bp-kelley/descriptastorus
pip install lmdb
pip install seaborn
pip install DeepPurpose
pip install wandb
pip install pydantic
conda install -c conda-forge pytdc
```

A version of torch 2.1+ is required to be installed since Jul requires a version of torch >=2.1.0. 

1. 1. If you want to use GPU, then first find a matched torch version, then install duel with cuda version. We give an example of torch 2.3.0 with cuda 11.8:
    ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   conda install -c dglteam/label/th23_cu118 dgl
   ```
2. If you are not using a GPU, then follow this:
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    conda install -c dglteam/label/th23_cpu dgl
    ```

### Tutorials 


## Example

We give two examples for each case study. One is trained with fixed parameters (a) and one is trained with argument. The argument list is given below.


| Argument  | Description                                     |
|-----------------|-------------------------------------------------|
| target_encoding             | 'CNN' / 'Transformer' for sequential learning, or 'DGL_GCN' for 'DGL_AttentiveFP' for structure learning. Current available protein encoding belongs to this full list: ['CNN', 'Transformer', 'CNN_RNN', 'DGL_GCN', 'DGL_GAT', 'DGL_AttentiveFP', 'DGL_NeuralFP', 'DGL_MPNN', 'PAGTN', 'Graphormer']  |
| seed         | For paper: 7 / 42 /100. You could try your own seed.            |
| wandb_proj     | The name of your wandb project that you wish to save the results into.                     |
| lr          | Learning rate. We recommend 1e-4 for non-GNN learning and 1e-5 for GNN learning                  |
| epochs         | Number of training epochs. Generally setting 60 - 100 epochs leads to convergence.                     |
| compute_pos_enc *    | Compute positional encoding for using graph transformers. We dont recommend add this inductive bias into GNN as GNN itself already encoded it. This don't work effectively on large scale graphs with dgl so the implementation is still under test.                           |
| batch_size | Batch size of 8 - 32 is good for protein sequence learning.                |




### Case Study 1(a): A Framework for Protein Function (Property) Prediction with sequential learning.
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os
import sys
import argparse
import torch
import wandb


### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.ProteinPred as models

### Load Beta lactamase dataset
path = os.getcwd()
train_fluo = Beta_lactamase(path + '/DeepProtein/data', 'train')
valid_fluo = Beta_lactamase(path + '/DeepProtein/data', 'valid')
test_fluo = Beta_lactamase(path + '/DeepProtein/data', 'test')

train_protein_processed, train_target, train_protein_idx = collate_fn(train_fluo)
valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_fluo)
test_protein_processed, test_target, test_protein_idx = collate_fn(test_fluo)

### Train Valid Test Split
target_encoding = 'CNN'
train, _, _ = utils.data_process(X_target=train_protein_processed, y=train_target, 
    target_encoding=target_encoding, split_method='random', frac=[0.99998, 1e-5, 1e-5])

_, val, _ = utils.data_process(X_target=valid_protein_processed, y=valid_target,        
    target_encoding=target_encoding, split_method='random', frac=[1e-5, 0.99998, 1e-5])

_, _, test = utils.data_process(X_target=test_protein_processed, y=test_target,         
    target_encoding=target_encoding,split_method='random', frac=[1e-5, 1e-5, 0.99998])
                            
### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[1024, 1024],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
config['multi'] = False
torch.manual_seed(args.seed)
model = models.model_initialize(**config)

### Train our model
model.train(train, val, test, compute_pos_enc = False)

```

</details>

If you want to use structure learning methods such as graph neural network, please set the second parameters in the collate_fn() into True.


### Protein-Protein Interaction (PPI)

```python
python train/ppi_affinity.py --target_encoding CNN --seed 7 --wandb_proj DeepPurposePP --lr 0.0001 --epochs 60
```


###  Protein Localization Prediction

###  Antigen Epitope Prediction

###  Antibody Paratope Prediction, 

###  Antibody Developability Prediction    

## Contact
Please contact jiaxie@ethz.ch or futianfan@gmail.com for help or submit an issue. 

## Encodings
Thanks to DeepPurpose and dgllife, we could borrow some of the encodings from DeepPurpose. The newly added encodings are PAGTN, 
EGT and Graphormer which belong to graph transformer modules that are prevailing methods
these years for encoding protein graphs.

Currently, we support the following encodings:

| Protein Encodings  | Description                                     |
|-----------------|-------------------------------------------------|
| CNN             | Convolutional Neural Network on SMILES          |
| CNN_RNN         | A GRU/LSTM on top of a CNN on SMILES            |
| Transformer     | Transformer Encoder on ESPF                     |
| MPNN            | Message-passing neural network                  |
| DGL_GCN         | Graph Convolutional Network                     |
| DGL_NeuralFP    | Neural Fingerprint                              |
| DGL_AttentiveFP | Attentive FP, Xiong et al. 2020                 |
| DGL_GAT         | Graph Attention Network                         |
| PAGTN           | Path Augmented Graph Transformer Network        |
| Graphormer      | Do Transformers Really Perform Bad, Ying et al. |


Note that we've tried EGT, however, it would lead to memory error if we want to 
construct a large batched edge feature matrix therefore we ignore the implementation of EGT.
This could be solved if applied to small graphs so it will be our future work. 




## Cite Us
If you found this package useful, please cite [our paper](https://doi.org/10.1093/bioinformatics/btaa1005):
```
@article{huang2020deeppurpose,
  title={DeepPurpose: A Deep Learning Library for Drug-Target Interaction Prediction},
  author={Huang, Kexin and Fu, Tianfan and Glass, Lucas M and Zitnik, Marinka and Xiao, Cao and Sun, Jimeng},
  journal={Bioinformatics},
  year={2020}
}
```

