import numpy as np
import torch
from .utils import *
from rdkit import Chem
import lmdb
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def dataset_factory(data_file: Union[str, Path], *args, **kwargs) -> Dataset:
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(data_file)
    if data_file.suffix == '.lmdb':
        return LMDBDataset(data_file, *args, **kwargs)
    else:
        raise ValueError(f"Unrecognized datafile type {data_file.suffix}")


import pickle as pkl


class LMDBDataset(Dataset):
    """Creates a dataset from an lmdb file.
    Args:
        data_file (Union[str, Path]): Path to lmdb file.
        in_memory (bool, optional): Whether to load the full dataset into memory.
            Default: False.
    """

    def __init__(self,
                 data_file: Union[str, Path],
                 in_memory: bool = False):

        data_file = Path(data_file)
        if not data_file.exists():
            raise FileNotFoundError(data_file)

        env = lmdb.open(str(data_file), max_readers=1, readonly=True,
                        lock=False, readahead=False, meminit=False)

        with env.begin(write=False) as txn:
            num_examples = pkl.loads(txn.get(b'num_examples'))

        if in_memory:
            cache = [None] * num_examples
            self._cache = cache

        self._env = env
        self._in_memory = in_memory
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)

        if self._in_memory and self._cache[index] is not None:
            item = self._cache[index]
        else:
            with self._env.begin(write=False) as txn:
                item = pkl.loads(txn.get(str(index).encode()))
                if 'id' not in item:
                    item['id'] = str(index)
                if self._in_memory:
                    self._cache[index] = item
        return item



class Subcellular(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'subcellular_localization/subcellular_localization_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        protein_orig = item['primary']
        target = item['localization']

        return protein_orig, target


class BinarySubcellular(Dataset):
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'subcellular_localization_2/subcellular_localization_2_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        protein_orig = item['primary']
        target = item['localization']

        return protein_orig, target


class FluorescenceDataset(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'fluorescence/fluorescence_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        protein_orig = item['primary']
        target = item['log_fluorescence'][0]

        return protein_orig, target


class Beta_lactamase(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'beta_lactamase/beta_lactamase_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_orig = item['primary']
        target = item['scaled_effect1']
        # print(protein_orig, target)
        return protein_orig, target




class Stability(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'stability/stability_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_orig = item['primary']
        target = item['stability_score']
        # print(protein_orig, target)
        return protein_orig, target

class Solubility(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'solubility/solubility_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]
        protein_orig = item['primary']
        target = item['solubility']
        # print(protein_orig, target)
        return protein_orig, target



### ----------------------- PPI ------------------------------- ###

class PPI_Affinity(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'ppi_affinity/ppi_affinity_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        graph1 = self.data[index]['primary_1']
        graph2 = self.data[index]['primary_2']
        target = self.data[index]['interaction']
        # print(protein_orig, target)
        return graph1, graph2, target

class HUMAN_PPI(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'human_ppi/human_ppi_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        graph1 = self.data[index]['primary_1']
        graph2 = self.data[index]['primary_2']
        target = self.data[index]['interaction']
        # print(protein_orig, target)
        return graph1, graph2, target

class Yeast_PPI(Dataset):

    def __init__(self,
                 data_path: Union[str, Path],
                 split: str,
                 in_memory: bool = False):
        if split not in ('train', 'valid', 'test'):
            raise ValueError(f"Unrecognized split: {split}. "
                             f"Must be one of ['train', 'valid', 'test']")

        data_path = Path(data_path)
        data_file = f'yeast_ppi/yeast_ppi_{split}.lmdb'
        self.data = dataset_factory(data_path / data_file, in_memory)
        # print(self.__getitem__(0))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        graph1 = self.data[index]['primary_1']
        graph2 = self.data[index]['primary_2']
        target = self.data[index]['interaction']
        # print(protein_orig, target)
        return graph1, graph2, target




def collate_fn(batch, graph=False, unsqueeze=True):
    # Unpack batch into protein sequences and targets
    protein_orig, target = tuple(zip(*batch))
    protein_orig = list(protein_orig)
    target = list(target)
    batch_len = len(target)

    # Create index array for protein sequences (original indices)
    protein_idx = np.array(list(range(batch_len)))

    if graph:
        # Lists to store valid proteins and targets
        valid_proteins = []
        valid_targets = []

        # Process each protein sequence
        for i in tqdm(range(batch_len)):
            # Try to convert protein sequence to a molecule and then to SMILES
            mol = Chem.MolFromSequence(protein_orig[i])

            if mol is not None:
                # Append valid protein and target
                valid_proteins.append(Chem.MolToSmiles(mol))
                valid_targets.append(target[i])
            else:
                # Log warning for invalid sequence
                print(f"Warning: Failed to create molecule from sequence: {protein_orig[i]}")
        # Use valid proteins and targets
        protein_orig = valid_proteins
        target = valid_targets

        # Create a new protein_idx with indices from 0 to valid protein count - 1
        protein_idx = np.array(list(range(len(valid_proteins))))

    # Convert target to torch tensor
    target = torch.FloatTensor(np.array(target))
    if unsqueeze:
        # Unsqueeze to add an extra dimension if needed
        target = target.unsqueeze(1)

    # Return processed protein sequences, target values, and the protein indices
    return protein_orig, target, protein_idx


def collate_fn_ppi(batch, graph=False, unsqueeze=True):
    # Unpack batch into protein sequences and targets
    graph1, graph2, target = tuple(zip(*batch))
    graph1 = list(graph1)
    graph2 = list(graph2)
    target = list(target)
    batch_len = len(target)

    # Create index array for protein sequences (original indices)
    protein_idx = np.array(list(range(batch_len)))

    if graph:
        # Lists to store valid proteins and targets
        valid_proteins1 = []
        valid_proteins2 = []
        valid_targets = []

        # Process each protein sequence
        for i in tqdm(range(batch_len)):
            # Try to convert protein sequence to a molecule and then to SMILES
            mol1 = Chem.MolFromSequence(graph1[i])
            mol2 = Chem.MolFromSequence(graph2[i])
            if mol1 is not None and mol2 is not None:
                # Append valid protein and target
                valid_proteins1.append(Chem.MolToSmiles(mol1))
                valid_proteins2.append(Chem.MolToSmiles(mol2))
                valid_targets.append(target[i])
            else:
                # Log warning for invalid sequence
                print(f"Warning: Failed to create molecule from sequence")
        # Use valid proteins and targets
        graph1 = valid_proteins1
        graph2 = valid_proteins2
        target = valid_targets

        # Create a new protein_idx with indices from 0 to valid protein count - 1
        protein_idx = np.array(list(range(len(valid_proteins))))

    # Convert target to torch tensor
    target = torch.FloatTensor(np.array(target))
    if unsqueeze:
        # Unsqueeze to add an extra dimension if needed
        target = target.unsqueeze(1)

    # Return processed protein sequences, target values, and the protein indices
    return graph1, graph2, target, protein_idx