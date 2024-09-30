<p align="center"><img src="figs/deeppurpose_pp_logo.png" alt="logo" width="400px" /></p>

<h3 align="center">
<p> DeepProtein: Deep Learning Library and Benchmark for Protein Sequence Learning <br></h3>
<h4 align="center">
<p>  Applications in Protein Property Prediction, Localization Prediction, Protein-Protein Interaction, antigen epitope prediction, antibody paratope prediction, antibody developability prediction, etc. </h4>




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

1. If you want to use GPU, then first find a matched torch version, then install duel with cuda version. We give an example of torch 2.3.0 with cuda 11.8:
    ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   conda install -c dglteam/label/th23_cu118 dgl
   ```
2. If you are not using a GPU, then follow this:
    ```bash
    pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
    conda install -c dglteam/label/th23_cpu dgl
    ```

## Demos

Checkout some demos & tutorials to start:

| Name | Description |
|-----------------|-------------|
| [Dataset Tutorial](DEMO/load_dataset.ipynb) | Tutorial on how to use the dataset loader and read customized data|
| [Protein Property Prediction Tutorial](DEMO/lactamase_Protein_Property.ipynb)| Example of CNN on Beta-lactamase property prediction|
| [Protein Protein Interaction Tutorial](DEMO/ppi_affinity_PPI.ipynb)| Example of CNN on PPI Affinity prediction|


## Example

We give two examples for each case study. One is trained with fixed parameters (a) and one is trained with argument. The argument list is given below.


| Argument  | Description                                     |
|-----------------|-------------------------------------------------|
| target_encoding             | 'CNN' / 'Transformer' for sequential learning, or 'DGL_GCN' for 'DGL_AttentiveFP' for structure learning. Current available protein encoding belongs to this full list: ['CNN', 'Transformer', 'CNN_RNN', 'DGL_GCN', 'DGL_GAT', 'DGL_AttentiveFP', 'DGL_NeuralFP', 'DGL_MPNN', 'PAGTN', 'Graphormer']. For residue level tasks, the protein encoding list is ['Token_CNN', 'Token_CNN_RNN, 'Token_Transformer']  |
| seed         | For paper: 7 / 42 /100. You could try your own seed.            |
| wandb_proj     | The name of your wandb project that you wish to save the results into.                     |
| lr          | Learning rate. We recommend 1e-4 for non-GNN learning and 1e-5 for GNN learning.                  |
| epochs         | Number of training epochs. Generally setting 60 - 100 epochs leads to convergence.                     |
| compute_pos_enc *    | Compute positional encoding for using graph transformers. We dont recommend add this inductive bias into GNN as GNN itself already encoded it. This don't work effectively on large scale graphs with dgl so the implementation is still under test.                           |
| batch_size | Batch size of 8 - 32 is good for protein sequence learning.                |




### Case Study 1(a): A Framework for Protein Function (Property) Prediction 
<details>
  <summary>Click here for the code!</summary>

```python
import os, sys, argparse, torch, wandb

### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.ProteinPred as models

### Load Beta lactamase dataset
path = os.getcwd()
train_beta = Beta_lactamase(path + '/DeepProtein/data', 'train')
valid_beta = Beta_lactamase(path + '/DeepProtein/data', 'valid')
test_beta = Beta_lactamase(path + '/DeepProtein/data', 'test')

train_protein_processed, train_target, train_protein_idx = collate_fn(train_beta)
valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_beta)
test_protein_processed, test_target, test_protein_idx = collate_fn(test_beta)

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

(b) If you wish to use arguments, this could be trained in one line. All mentioned GNN variants above is available for training.

<details>
  <summary>CNN Case</summary>

```python 
python train/beta.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>
<details>
  <summary>GNN Case</summary>

```python 
python train/beta.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100
```

</details>

### Case Study 1(b): A Framework for Protein Protein Interaction Prediction 
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb

### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.PPI as models

### Load PPI Affinity dataset
path = os.getcwd()
train_ppi = PPI_Affinity(path + '/DeepProtein/data', 'train')
valid_ppi = PPI_Affinity(path + '/DeepProtein/data', 'valid')
test_ppi = PPI_Affinity(path + '/DeepProtein/data', 'test')

train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train_ppi, graph=False, unsqueeze=False)
valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid_ppi, graph=False, unsqueeze=False)
test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_ppi, graph=False, unsqueeze=False)

### Train Valid Test Split
target_encoding = 'CNN'
train, _, _ = data_process(X_target = train_protein_1, X_target_ = train_protein_2, y = train_target,
                target_encoding = target_encoding,
                split_method='random', frac=[0.99998, 1e-5, 1e-5],
                random_seed = 1)
_, val, _ = data_process(X_target = valid_protein_1, X_target_ = valid_protein_2, y = valid_target,
                target_encoding = target_encoding,
                split_method='random',frac=[1e-5, 0.99998, 1e-5],
                random_seed = 1)

_, _, test = data_process(X_target = test_protein_1, X_target_ = test_protein_2, y = test_target,
                target_encoding = target_encoding,
                split_method='random',frac=[1e-5, 1e-5, 0.99998],
                random_seed = 1)
                            
### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[512],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
# config['multi'] = False
torch.manual_seed(args.seed)
model = models.model_initialize(**config)

### Train our model
model.train(train, val, test)

```
</details>

If you want to use structure learning methods such as graph neural network, please set the second parameters in the collate_fn_ppi() into True.

(b) If you wish to use arguments, this could be trained in one line. For GNN, only DGL_GCN, DGL_GAT and DGL_NeuralFP is available currently.


<details>
  <summary>CNN Case</summary>

```python 
python train/ppi_affinity.py --target_encoding CNN --seed 42 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>
<details>
  <summary>GNN Case</summary>

```python 
python train/ppi_affinity.py --target_encoding DGL_GCN --seed 42 --wandb_proj DeepProtein --lr 0.00001 --epochs 100
```

</details>



### Case Study 1(c): A Framework for Protein Localization Prediction 
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb

### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.ProteinPred as models

### Load Subcellular Dataset
path = os.getcwd()
train_sub = Subcellular(path + '/DeepProtein/data', 'train')
valid_sub = Subcellular(path + '/DeepProtein/data', 'valid')
test_sub = Subcellular(path + '/DeepProtein/data', 'test')

train_protein_processed, train_target, train_protein_idx = collate_fn(train_sub)
valid_protein_processed, valid_target, valid_protein_idx = collate_fn(valid_sub)
test_protein_processed, test_target, test_protein_idx = collate_fn(test_sub)

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
config['binary'] = False
config['multi'] = True
config['classes'] = 10
torch.manual_seed(args.seed)
model = models.model_initialize(**config)

### Train our model
model.train(train, val, test, compute_pos_enc = False)

```

</details>

If you want to use structure learning methods such as graph neural network, please set the second parameters in the collate_fn() into True. Note that SubCellular is multi-class classification problem, therefore you should set config['multi'] to True.

(b) If you wish to use arguments, this could be trained in one line. All mentioned GNN variants above is available for training.

<details>
  <summary>CNN Case</summary>

```python 
python train/subcellular.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>
<details>
  <summary>GNN Case</summary>

```python 
python train/subcellular.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100
```
</details>

### Case Study 1(d): A Framework for Antigen Epitope Prediction
Make sure that tdc is installed, if not 
```bash
pip install PyTDC
```
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb

### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.TokenPred as models
from tdc.single_pred import Epitope

### Load Epitope  Dataset
data_class, name, X = Epitope, 'IEDB_Jespersen', 'Antigen'
data = data_class(name=name)
split = data.get_split()

train_data, valid_data, test_data = split['train'], split['valid'], split['test']
vocab_set = set()

train_vocab, train_positive_ratio = data2vocab(train_data, train_data, X)
valid_vocab, valid_positive_ratio = data2vocab(valid_data, train_data, X)
test_vocab, test_positive_ratio = data2vocab(test_data, train_data, X)

vocab_set = train_vocab.union(valid_vocab)
vocab_set = vocab_set.union(test_vocab)
vocab_lst = list(vocab_set)

### Train Valid Test Split
train_data = standardize_data(train_data, vocab_lst, X)
valid_data = standardize_data(valid_data, vocab_lst, X)
test_data = standardize_data(test_data, vocab_lst, X)

train_set = data_process_loader_Token_Protein_Prediction(train_data)
valid_set = data_process_loader_Token_Protein_Prediction(valid_data)
test_set = data_process_loader_Token_Protein_Prediction(test_data)

### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[1024, 1024],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
config['multi'] = False
config['binary'] = True
config['token'] = True
config['in_channels'] = 24
torch.manual_seed(args.seed)
model = models.model_initialize(**config)


### Train our model
model.train(train_set, valid_set, test_set, batch_size=batch_size)

```

</details>



(b) If you wish to use arguments, this could be trained in one line. All mentioned GNN variants above is not available for training. For an important notice, this task is residue (token) level classification therefore we deploy token level CNN, CNN_RNN and Transformer models.

Current version only supports Token_CNN, Token_CNN_RNN and Token_Transformer for target_encoding.

<details>
  <summary>CNN Case</summary>

```python 
python train/IEDB.py --target_encoding Token_CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>

### Case Study 1(e): A Framework for Antibody Paratope Prediction
Make sure that tdc is installed, if not 
```bash
pip install PyTDC
```
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb 

### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.TokenPred as models
from tdc.single_pred import Paratope

### Load Paratope Dataset
data_class, name, X = Paratope, 'SAbDab_Liberis', 'Antibody'
data = data_class(name=name)
split = data.get_split()
train_data, valid_data, test_data = split['train'], split['valid'], split['test']
vocab_set = set()

train_vocab, train_positive_ratio = data2vocab(train_data, train_data, X)
valid_vocab, valid_positive_ratio = data2vocab(valid_data, train_data, X)
test_vocab, test_positive_ratio = data2vocab(test_data, train_data, X)

vocab_set = train_vocab.union(valid_vocab)
vocab_set = vocab_set.union(test_vocab)
vocab_lst = list(vocab_set)



### Train Valid Test Split
train_data = standardize_data(train_data, vocab_lst, X)
valid_data = standardize_data(valid_data, vocab_lst, X)
test_data = standardize_data(test_data, vocab_lst, X)

train_set = data_process_loader_Token_Protein_Prediction(train_data)
valid_set = data_process_loader_Token_Protein_Prediction(valid_data)
test_set = data_process_loader_Token_Protein_Prediction(test_data)

### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[1024, 1024],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
config['multi'] = False
config['binary'] = True
config['token'] = True
config['in_channels'] = 20
torch.manual_seed(args.seed)
model = models.model_initialize(**config)


### Train our model
model.train(train_set, valid_set, test_set, batch_size=batch_size)

```

</details>

 

(b) If you wish to use arguments, this could be trained in one line. All mentioned GNN variants above is not available for training. For an important notice, this task is residue (token) level classification therefore we deploy token level CNN, CNN_RNN and Transformer models.

Current version only supports Token_CNN, Token_CNN_RNN and Token_Transformer for target_encoding.

<details>
  <summary>CNN Case</summary>

```python 
python train/SAbDab_Liberis.py --target_encoding Token_CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>


### Case Study 1(f): A Framework for Antibody Developability Prediction (TAP)
Make sure that tdc is installed, if not 
```bash
pip install PyTDC
```
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb


### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.PPI as models
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop

### Load TAP Dataset
label_list = retrieve_label_name_list('TAP')

data = Develop(name='TAP', label_name=label_list[0])
split = data.get_split()

train_antibody_1, train_antibody_2 = to_two_seq(split, 'train', 'Antibody')
valid_antibody_1, valid_antibody_2 = to_two_seq(split, 'valid', 'Antibody')
test_antibody_1, test_antibody_2 = to_two_seq(split, 'test', 'Antibody')

y_train, y_valid, y_test = split['train']['Y'], split['valid']['Y'], split['test']['Y']

train_TAP = list(zip(train_antibody_1, train_antibody_2, y_train))
valid_TAP = list(zip(valid_antibody_1, valid_antibody_2, y_valid))
test_TAP = list(zip(test_antibody_1, test_antibody_2, y_test))


### Train Valid Test Split
target_encoding = 'CNN'
train_protein_1, train_protein_2, train_target, train_protein_idx = collate_fn_ppi(train_TAP, graph=False, unsqueeze=False)
valid_protein_1, valid_protein_2, valid_target, valid_protein_idx = collate_fn_ppi(valid_TAP, graph=False, unsqueeze=False)
test_protein_1, test_protein_2, test_target, test_protein_idx = collate_fn_ppi(test_TAP, graph=False, unsqueeze=False)

train, _, _ = data_process(X_target=train_protein_1, X_target_=train_protein_2, y=train_target,
                               target_encoding=target_encoding,
                               split_method='random', frac=[0.99998, 1e-5, 1e-5],
                               random_seed=1)

_, val, _ = data_process(X_target=valid_protein_1, X_target_=valid_protein_2, y=valid_target,
                            target_encoding=target_encoding,
                            split_method='random', frac=[1e-5, 0.99998, 1e-5],
                            random_seed=1)

_, _, test = data_process(X_target=test_protein_1, X_target_=test_protein_2, y=test_target,
                            target_encoding=target_encoding,
                            split_method='random', frac=[1e-5, 1e-5, 0.99998],
                            random_seed=1)

### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[1024, 1024],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
config['binary'] = False
config['multi'] = False
torch.manual_seed(args.seed)
model = models.model_initialize(**config)


### Train our model
model.train(train_set, valid_set, test_set, batch_size=batch_size)

```

</details>



If you want to use structure learning methods such as graph neural network, please set the second parameters in the collate_fn_ppi() into True.

(b) If you wish to use arguments, this could be trained in one line. For GNN, only DGL_GCN, DGL_GAT and DGL_NeuralFP is available currently.

<details>
  <summary>CNN Case</summary>

```python 
python train/TAP.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>

<details>
  <summary>GNN Case</summary>

```python 
python train/TAP.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100
```

</details>


### Case Study 1(g): A Framework for Repair Outcome Prediction (CRISPR)

Make sure that tdc is installed, if not 
```bash
pip install PyTDC
```
<details>
  <summary>Click here for the code!</summary>

```python
### package import
import os, sys, argparse, torch, wandb


### Our library DeepProtein
from DeepProtein.dataset import *
import DeepProtein.utils as utils
import DeepProtein.ProteinPred as models
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop, CRISPROutcome

### Load CRISPR Leenay Dataset
label_list = retrieve_label_name_list('Leenay')

data = CRISPROutcome(name='Leenay', label_name=label_list[0])
split = data.get_split()

train_GuideSeq, y_train = list(split['train']['GuideSeq']), list(split['train']['Y'])
val_GuideSeq, y_valid = list(split['valid']['GuideSeq']), list(split['valid']['Y'])
test_GuideSeq, y_test = list(split['test']['GuideSeq']), list(split['test']['Y'])

train_CRISPR = list(zip(train_GuideSeq, y_train))
valid_CRISPR = list(zip(val_GuideSeq, y_valid))
test_CRISPR = list(zip(test_GuideSeq, y_test))


### Train Valid Test Split
target_encoding = 'CNN'
train_protein_1, train_target, train_protein_idx = collate_fn(train_CRISPR, graph=False, unsqueeze=True)
valid_protein_1, valid_target, valid_protein_idx = collate_fn(valid_CRISPR, graph=False, unsqueeze=True)
test_protein_1, test_target, test_protein_idx = collate_fn(test_CRISPR, graph=False, unsqueeze=True)

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

### Load configuration for model
config = generate_config(target_encoding=target_encoding,
                         cls_hidden_dims=[1024, 1024],
                         train_epoch=20,
                         LR=0.0001,
                         batch_size=32,
                         )
config['binary'] = False
config['multi'] = False
torch.manual_seed(args.seed)
model = models.model_initialize(**config)


### Train our model
model.train(train_set, valid_set, test_set, batch_size=batch_size)

```

</details>



If you want to use structure learning methods such as graph neural network, please set the second parameters in the collate_fn() into True.

(b) If you wish to use arguments, this could be trained in one line. All mentioned GNN above is available currently.

<details>
  <summary>CNN Case</summary>

```python 
python train/CRISPR.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100
```

</details>

<details>
  <summary>GNN Case</summary>

```python 
python train/CRISPR.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100
```

</details>

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

## Data
We provided the data under the folder DeepProtein/data (Besides TDC) and the folder data (TDC). 
| Data  | Source | Form | Task |
|----------|----------|----------|----------|
| Fluorescence | PEER | LMDB | Protein Property Prediction |
| Solubility | PEER | LMDB | Protein Property Prediction |
| Stability | PEER | LMDB | Protein Property Prediction |
| Beta-lactamase | PEER | LMDB | Protein Property Prediction |
| SubCellular | PEER | LMDB | Protein Localization Prediction | 
| SubCellular-Binary | PEER | LMDB | Protein Localization Prediction | 
| PPI_Affinity | PEER | LMDB | Protein-Protein Interaction |
| Human_PPI | PEER | LMDB | Protein-Protein Interaction |
| Yeast_PPI | PEER | LMDB | Protein-Protein Interaction |
| IEDB | TDC | PICKLE | Antigen Epitope Prediction  |
| PDB-Jespersen | TDC | PICKLE | Antigen Epitope Prediction |
| SAbDab-Liberis | TDC | PICKLE | Antibody Paratope Prediction |
| TAP | TDC | TAB | Antibody Developability Prediction  |
| SAbDab-Chen | TDC | TAB | Antibody Developability Prediction |
| CRISPR-Leenay | TDC | TAB | CRISPR Repair Outcome Prediction |
