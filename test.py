from DeepPurpose_PP.dataset import *
import os
import DeepPurpose_PP.utils as utils




if __name__ == "__main__":

    encoding = "CNN"

    path = os.getcwd()
    # 1. Test on FluorescenceDataset
    train_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'valid')
    train_protein_processed, train_target, train_protein_idx  = collate_fn(train_fluo)


    train, val, test = utils.data_process(X_drug = train_protein_processed, y = train_target, drug_encoding = encoding,
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)
    

    