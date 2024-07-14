from DeepPurpose_PP.dataset import *
import os
import DeepPurpose_PP.utils as utils
import DeepPurpose_PP.ProteinPred as models



if __name__ == "__main__":


    target_encoding = "CNN"

    # path = os.getcwd()
    path = "/itet-stor/jiaxie/net_scratch/DeepPurposePlusPlus"
    #  Test on FluorescenceDataset
    train_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'train')
    train_protein_processed, train_target, train_protein_idx  = collate_fn(train_fluo)

    valid_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'valid')
    valid_protein_processed, valid_target, valid_protein_idx  = collate_fn(valid_fluo)

    test_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'test')
    test_protein_processed, test_target, test_protein_idx  = collate_fn(test_fluo)

    train, _, _ = utils.data_process(X_target = train_protein_processed, y = train_target, target_encoding = target_encoding, 
                                        # drug_encoding= drug_encoding,
                                split_method='random',frac=[0.9998,1e-4,1e-4],
                                random_seed = 1)
    
    _, val, _ = utils.data_process(X_target = valid_protein_processed, y = valid_target, target_encoding = target_encoding, 
                                        # drug_encoding= drug_encoding,
                                split_method='random',frac=[1e-4,0.9998,1e-4],
                                random_seed = 1)
    
    _, _, test = utils.data_process(X_target = test_protein_processed, y = test_target, target_encoding = target_encoding, 
                                        # drug_encoding= drug_encoding,
                                split_method='random',frac=[1e-4,1e-4,0.9998],
                                random_seed = 1)


    config = generate_config(target_encoding = target_encoding, 
                         cls_hidden_dims = [512], 
                         train_epoch = 20, 
                         LR = 0.0008, 
                         batch_size = 128,
                        )

    model = models.model_initialize(**config)
    model.train(train, val, test)


    