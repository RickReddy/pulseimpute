from configs.test_transformer_configs import *
from configs.test_brits_configs import *
from configs.test_other_configs import *
from configs.test_naomi_configs import *

from utils.evaluate_imputation import eval_mse, eval_heartbeat_detection, eval_cardiac_classification, printlog
from utils.loss_mask import mse_mask_loss
from ast import literal_eval


import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm

import faulthandler

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars


if __name__=='__main__':

    configs = [bdc883_emb256_layer2_hcpa_ppg_test, bdc883_emb256_layer2_hcpya_ppg_test, 
    bdc883_emb256_layer2_nki_ppg_test, bdc883_emb256_layer2_eegfmri_vu_ppg_test, 
    bdc883_emb256_layer2_eegfmri_nih_ppg_test]

    for config in configs:
        print(config["modelname"]+config["annotate"]+config["annotate_test"])
        faulthandler.enable()

        random_seed(10, True)
        load = getattr(__import__(f'utils.{config["data_name"]}', fromlist=['']), "load")
        X_train, Y_dict_train, X_val, Y_dict_val, X_test, Y_dict_test = load(**config["data_load"])
        #print(Y_dict_test)
        np.save(os.path.join("imputations", config["data_name"]+config["annotate_test"]+'_raw_ytest'), Y_dict_test['target_seq'])

        path = os.path.join("out/", config["data_name"]+config["annotate_test"], config["modelname"]+config["annotate"])
        model_type = config["modeltype"]
        model_module = __import__(f'models.{model_type}_model', fromlist=[''])
        model_module_class = getattr(model_module, model_type)
        model = model_module_class(modelname=config["modelname"], data_name=config["data_name"], 
                                train_data=X_test, val_data=None, imputation_dict=Y_dict_test,
                                annotate=config["annotate"],  annotate_test=config["annotate_test"],  
                                **config["modelparams"],
                                **config["train"])
        imputation = model.testimp()

        np.save(os.path.join("vis_output", config["data_name"]+config["annotate_test"]+'_imputation'), imputation)
        
        mse_all_data = []
        for i in range(len(imputation)):
            mse_scan_specific = []
            ppgmiss_idx = i % len(Y_dict_test['list_of_missing'])
            #print(list_of_miss)
            running_index = 0
            
            for j in range(len(Y_dict_test['list_of_missing'][ppgmiss_idx])):
                list_of_miss = literal_eval(Y_dict_test['list_of_missing'][ppgmiss_idx][j])
                #print(list_of_miss[j])
        
                #print(mse_mask_loss(Y_dict_test['target_seq'][i], imputation[i]))
                if list_of_miss[0] == 0:
                    ground_truth = Y_dict_test['target_seq'][i][running_index:running_index+list_of_miss[1]]
                    #ground_truth = torch.from_numpy(Y_dict_test['target_seq'][i][running_index:running_index+list_of_miss[1]])
                    imputed = torch.from_numpy(imputation[i][running_index:running_index+list_of_miss[1]])
                    mse_loss, missing_length = mse_mask_loss(ground_truth, imputed)
                    mse_scan_specific.append((mse_loss/missing_length).item())
                running_index += list_of_miss[1]
            mse_all_data.append(mse_scan_specific)
        df = pd.DataFrame(mse_all_data)
        df.to_csv(os.path.join("vis_output", config["data_name"] + "_MSE.csv"), index=False, header=False)