from DeepPurpose_PP.dataset import *
import os
from  DeepPurpose_PP import utils




if __name__ == "__main__":

    path = os.getcwd()
    # 1. Test on FluorescenceDataset
    train_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'valid')
    train_batch = collate_fn(train_fluo)


    train, val, test = utils.data_process(X_drug = X_drugs, y = y, drug_encoding = drug_encoding,
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    

    