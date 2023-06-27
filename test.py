from DeepPurpose_PP.dataset import *
import os
import DeepPurpose_PP.utils as utils
import DeepPurpose_PP.ProteinPred as models



if __name__ == "__main__":



    target_encoding = "CNN"

    path = os.getcwd()
    # 1. Test on FluorescenceDataset
    train_fluo = FluorescenceDataset(path + '/DeepPurpose_PP/data', 'valid')
    train_protein_processed, train_target, train_protein_idx  = collate_fn(train_fluo)


    train, val, test = utils.data_process(X_target = train_protein_processed, y = train_target, target_encoding = target_encoding, 
                                        # drug_encoding= drug_encoding,
                                split_method='random',frac=[0.7,0.1,0.2],
                                random_seed = 1)

    print(train[:1])


    config = generate_config(target_encoding = target_encoding, 
                         cls_hidden_dims = [512], 
                         train_epoch = 40, 
                         LR = 0.0008, 
                         batch_size = 128,
                        )
    
    
    model = models.model_initialize(**config)
    model.train(train, val, test)
