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



### Case Study 1(a): A Framework for Protein Function (Property) Prediction with sequential learning.
<details>
  <summary>Click here for the code!</summary>

```python
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *

SAVE_PATH='./saved_path'
import os 
if not os.path.exists(SAVE_PATH):
  os.makedirs(SAVE_PATH)


# Load Data, an array of SMILES for drug, an array of Amino Acid Sequence for Target and an array of binding values/0-1 label.
# e.g. ['Cc1ccc(CNS(=O)(=O)c2ccc(s2)S(N)(=O)=O)cc1', ...], ['MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTH...', ...], [0.46, 0.49, ...]
# In this example, BindingDB with Kd binding score is used.
X_drug, X_target, y  = process_BindingDB(download_BindingDB(SAVE_PATH),
					 y = 'Kd', 
					 binary = False, 
					 convert_to_log = True)

# Type in the encoding names for drug/protein.
drug_encoding, target_encoding = 'CNN', 'Transformer'

# Data processing, here we select cold protein split setup.
train, val, test = data_process(X_drug, X_target, y, 
                                drug_encoding, target_encoding, 
                                split_method='cold_protein', 
                                frac=[0.7,0.1,0.2])

# Generate new model using default parameters; also allow model tuning via input parameters.
config = generate_config(drug_encoding, target_encoding, transformer_n_layer_target = 8)
net = models.model_initialize(**config)

# Train the new model.
# Detailed output including a tidy table storing validation loss, metrics, AUC curves figures and etc. are stored in the ./result folder.
net.train(train, val, test)

# or simply load pretrained model from a model directory path or reproduced model name such as DeepDTA
net = models.model_pretrained(MODEL_PATH_DIR or MODEL_NAME)

# Repurpose using the trained model or pre-trained model
# In this example, loading repurposing dataset using Broad Repurposing Hub and SARS-CoV 3CL Protease Target.
X_repurpose, drug_name, drug_cid = load_broad_repurposing_hub(SAVE_PATH)
target, target_name = load_SARS_CoV_Protease_3CL()

_ = models.repurpose(X_repurpose, target, net, drug_name, target_name)

# Virtual screening using the trained model or pre-trained model 
X_repurpose, drug_name, target, target_name = ['CCCCCCCOc1cccc(c1)C([O-])=O', ...], ['16007391', ...], ['MLARRKPVLPALTINPTIAEGPSPTSEGASEANLVDLQKKLEEL...', ...], ['P36896', 'P00374']

_ = models.virtual_screening(X_repurpose, target, net, drug_name, target_name)

```

</details>




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

