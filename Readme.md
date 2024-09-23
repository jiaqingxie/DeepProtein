<p align="center"><img src="figs/deeppurpose_pp_logo.png" alt="logo" width="400px" /></p>

<h3 align="center">
<p> ProB: Deep Learning Library for Protein Sequence Learning <br></h3>
<h4 align="center">
<p>  Applications in Protein Property Prediction, Localization Prediction, Protein-Protein Interaction, antigen epitope prediction, antibody paratope prediction, antibody developability prediction, etc. </h4>

---

[![PyPI version](https://badge.fury.io/py/ProB.svg)](https://pypi.org/project/ProB/)
[![Downloads](https://pepy.tech/badge/ProB/month)](https://pepy.tech/project/ProB)
[![Downloads](https://pepy.tech/badge/ProB)](https://pepy.tech/project/ProB)
[![GitHub Repo stars](https://img.shields.io/github/stars/jiaqingxie/DeepPurposePlusPlus)](https://github.com/jiaqingxie/DeepPurposePlusPlus/stargazers)
[![GitHub Repo forks](https://img.shields.io/github/forks/jiaqingxie/DeepPurposePlusPlus)](https://github.com/jiaqingxie/DeepPurposePlusPlus/network/members)

### Installation

First we recommend you to follow the instruction of how DeepPurpose's dependencies are installed.
```bash
conda create -n prob python=3.9
conda activate prob
pip install git+https://github.com/bp-kelley/descriptastorus
pip install lmdb
pip install seaborn
pip install DeepPurpose
pip install wandb
pip install pydantic
conda install -c conda-forge pytdc
```

A version of torch 2.1+  is required to installed, since dgl required a version of torch >=2.1.0. 

1. If you want to use GPU, then firstly find a matched torch version, then install dgl with cuda version. We give an example of torch 2.3.0 with cuda 11.8:
    ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   conda install -c dglteam/label/th23_cu118 dgl
   ```
2. If you are not using a GPU, then follow this:
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    conda install -c dglteam/label/th23_cpu dgl
    ```



### Encodings

Thanks to DeepPurpose and dgllife, we could borrow some of the encodings from DeepPurpose. The new added encodings are PAGTN, 
EGT and Graphormer which belong to graph transformer modules that are prevailing methods
these years for encoding protein graphs.

Currently, we support the following encodings:

| Drug Encodings  | Description                                     |
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


###  Protein Function (Property) Prediction


Done: 
1. Dataset Built for Fluorescence Dataset 

test on the CNN probing on the Fluorescence dataset

```python
python CNN_Fluroscence.py
```

test on the GCN probing on the Fluorescence dataset

```python
python GCN_Fluroscence.py
```


```python
python test.py
```


### Protein-Protein Interaction (PPI)

```python
python train/ppi_affinity.py --target_encoding CNN --seed 7 --wandb_proj DeepPurposePP --lr 0.0001 --epochs 60
```

###  Protein Localization Prediction

###  Antigen Epitope Prediction

###  Antibody Paratope Prediction, 

###  Antibody Developability Prediction




