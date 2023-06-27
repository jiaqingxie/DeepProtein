
import numpy as np

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

def collate_fn(batch):
    protein_orig, target = tuple(zip(*batch))
    protein_orig = list(protein_orig)
    batch_len = len(target)


    protein_idx =  np.array(list(range(batch_len)))

    # protein_processed = []
    # for i in tqdm(range(batch_len)):
    #     protein_processed.append(Chem.MolToSmiles(Chem.MolFromSequence(protein_orig[i]))) 
        # if i % 10 == 0:
        #     print(protein_orig[i])

    
    # protein_processed =  [  for i in range(batch_len)]

    target = torch.FloatTensor(target)  # type: ignore
    target = target.unsqueeze(1)

    return protein_orig, target, protein_idx

if __name__ == "__main__":
    import os
    path = os.getcwd()
    # 1. Test on FluorescenceDataset
    train_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'train')
    valid_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'valid')
    test_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'test')


    # 2. Test on Processed Proteins
    train_batch = collate_fn(train_fluo)
    valid_batch = collate_fn(valid_fluo)
    test_batch = collate_fn(test_fluo)

    train_protein_processed, train_target, train_protein_idx = train_batch
    # print(train_target[:10])



