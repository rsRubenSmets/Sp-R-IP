import os
import copy
import sys
sys.path.insert(0,'../training')
import math
import random
import pickle
import functions_support_new as sf
import functions_cvxpylayers_new as clf
import torch_classes_new as tc



if __name__ == '__main__':


    ##### USER DECISIONS #####

    # Definition of the model
    reforecast_type = "ID_Q"  # This script only supports the ID_Q reforecaster
    nn_type = "softplus"  # "linear" or "softplus"
    warm_start = False

    #Definition of hyperaparmeters
    dict_hps = {
        'reg': [0.0], #[0,0.01],
        'batches': [64], #[8,64],
        'gamma': [3], #[0.1,0.3,1,3,10],
        'lr': [0.01], #[0.000005,0.00005],
        'loss_fct_str': ['profit'], #'profit', 'mse_sched' or 'mse_sched_weighted'
        'smoothing': ['logBar'], #'logBar' or 'quadratic'
        'framework': ["ID"], #"ID" (Implicit Differentiation) or "GS" (Gradient Smoothing) or "GS_proxy" (Gradient smoothing with proxy for mu calculation)
        'warm_start': [True,False]
        #'list_units': [[]],
        #'list_act': [[]] #[['softplus']]
    }

    #Folder to store results in
    store_code='20231120_test' #has to be other name than existing folder with results; best to include "_ID" in name for processing results

    ###### DEFINITION OF FIXED SETTINGS #####

    mkdir = False
    par_exec = False
    num_cpus = 1

    #Data split
    train_share = 1
    days_train = math.floor(5/train_share) #64
    last_ex_test = 59 #59
    repitition = 1

    factor_size_ESS = 1
    la=24

    training_dict = {
        'device': 'cpu',
        'model_type': "LR",
        'epochs': 2,
        'patience': 25,
        'type_train_labels': 'price_schedule',  # 'price' or 'price_schedule'
        'sched_type': 'net_discharge', #What type of schedule to use as labels: "net_discharge" or "extended" (latter being stack of d,c,soc)
        #'start_from_fc': warm_start,
    }

    data_dict = {
    #Dict containing all info required to retrieve and handle data
        'loc_data': '../data/processed_data/SPO_DA/',
        'feat_cols':["weekday", "NL+FR","GEN_FC","y_hat"],
        'col_label_price': 'y',
        'col_label_fc_price': 'y_hat',
        'lookahead': la,
        'days_train': days_train,
        'last_ex_test': last_ex_test,
        'train_share': train_share,
        'val_split_mode': 'alt_test', #'separate' for the validation set right before test set, 'alernating' for train/val examples alternating or 'alt_test' for val/test examples alternating
        'scale_mode': 'stand',  # 'norm','stand' or 'none'
        'scale_base': 'y_hat', #False or name of column; best to use in combination with scale_mode = 'stand'
        'cols_no_centering': ['y_hat'],
        'scale_price': True,
        'features_mu_calculator': ['price'],
        'labels_mu_calculator': ['mu']
    }


    OP_params_dict = {
    #Dict containing info of optimization program
        'max_charge': 0.01 * factor_size_ESS,
        'max_discharge': 0.01 * factor_size_ESS,
        'eff_d': 0.95,
        'eff_c': 0.95,
        'max_soc': 0.04 * factor_size_ESS,
        'min_soc': 0,
        'soc_0': 0.02 * factor_size_ESS,
        'ts_len': 1,
        'opti_type': 'exo',
        'opti_package': 'scipy',
        'lookahead': la,
        'soc_update': False,
        'cyclic_bc': True,
        'combined_c_d_max': False, #If True, c+d<= P_max; if False: c<=P_max; d<= P_max
        'degradation': False,
        'inv_cost': 0,
        'lookback': 0,
        'quantiles_FC_SI': [0.01, 0.05],
        'col_SI': 1,
        'perturbation': 0.2,
        'feat_cols': data_dict['feat_cols'],
        'restrict_forecaster_ts': True,
        'n_diff_features': len(data_dict['feat_cols']),
        #'smoothing': 'quadratic'  # 'quadratic' or 'log-barrier'
    }


    OP_params_dict['feat_cols'] = data_dict['feat_cols']
    OP_params_dict['n_diff_features'] =  len(data_dict['feat_cols'])

    #Set the parameters for the neural network architecture based on the choice of linear vs softplus network
    if nn_type == "linear":
        dict_hps['list_units'] = [[]]
        dict_hps['list_act'] = [[]]
    elif nn_type == "softplus":
        dict_hps['list_units'] = [[24]] # current implementation requires a single amount of hidden layers per run
        dict_hps['list_act'] = [['softplus']]


    ##### LOADING DATA, INITIALIZING MODEL AND TRAINING #####

    #Load data and split features/labels in numpy arrays
    features_train, features_val, features_test, price_train, price_val, price_test, price_fc_list,price_fc_scaled_list = sf.preprocess_data(data_dict)

    #Get optimal schedules if required
    labels_train,labels_val,labels_test = sf.preprocess_labels(training_dict, price_train, price_val, price_test, OP_params_dict, data_dict)

    #Convert to tensors
    list_tensors = sf.get_train_validation_tensors(features_train, labels_train, features_val, labels_val,features_test,labels_test,training_dict["model_type"],training_dict['type_train_labels'])


    #Creat a differentiable optimization layer. RN refers to Risk Neutral, meaning the original problem without smoothing term
    # This requires gamma to be 0 and smoothing to be defined at this point
    OP_params_dict['gamma'] = 0
    OP_params_dict['smoothing'] = 'quadratic'
    optiLayer_RN = tc.OptiLayer(OP_params_dict)

    training_dict['optiLayer_RN'] = optiLayer_RN

    # Create list to store all input information for the different HP combinations
    list_input_dicts = []
    hp_config = 0

    for ws in dict_hps['warm_start']:
        for reg in dict_hps['reg']:
            for bs in dict_hps['batches']:
                for gamma in dict_hps['gamma']:
                    for lr in dict_hps['lr']:
                        for list_units in dict_hps['list_units']:
                            for list_act in dict_hps['list_act']:
                                for loss in dict_hps['loss_fct_str']:
                                    for sm in dict_hps['smoothing']:
                                        for fw in dict_hps['framework']:
                                            for _ in range(repitition):
                                                hp_config+=1
                                                training_dict_local = copy.copy(training_dict)
                                                params_dict_local = copy.deepcopy(OP_params_dict)
                                                training_dict_local['start_from_fc'] = ws
                                                training_dict_local['reg'] = reg
                                                training_dict_local['batch_size'] = bs
                                                params_dict_local['gamma'] = gamma
                                                training_dict_local['lr'] = lr
                                                training_dict_local['list_units'] = list_units
                                                training_dict_local['list_act'] = list_act
                                                training_dict_local['loss_fct_str'] = loss
                                                training_dict_local['smoothing'] = sm
                                                training_dict_local['framework'] = fw


                                                input_dict={
                                                    'list_tensors': list_tensors,
                                                    'training_dict': training_dict_local,
                                                    'params_dict': params_dict_local,
                                                    'hp_config': hp_config
                                                }

                                                list_input_dicts.append(input_dict)



    #Create dir to store output
    store_code = f'../training/train_output/{store_code}/'
    if mkdir:
        os.makedirs(store_code)

    # #Train NN for all HP configurations
    # list_output_dicts = []
    # for input_dict in list_input_dicts:
    #     output_dict = clf.hp_tuning_cvxpylayers_2(input_dict)
    #     list_output_dicts.append(output_dict)

    from concurrent.futures import ProcessPoolExecutor
    import os


    def parallel_execution(input_dicts):
        list_output_dicts = []
        with ProcessPoolExecutor() as executor:
            list_output_dicts = list(executor.map(clf.hp_tuning_cvxpylayers_2, input_dicts))
        return list_output_dicts


    if par_exec:
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            list_output_dicts = parallel_execution(list_input_dicts)
    else:
        list_output_dicts = []
        for input_dict in list_input_dicts:
            out = clf.hp_tuning_cvxpylayers_2(input_dict)
            list_output_dicts.append(out)



    #Save output
    sf.save_outcome(list_output_dicts,store_code)





