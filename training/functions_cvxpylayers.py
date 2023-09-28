import torch
import functions_support as sf
import classes_torch as ct
import numpy as np
import time
import random
import copy

def hp_tuning_cvxpylayers_2(input_dict):
    #Overarching function of the training process, unpacking all the relevant data of the input dict, calling the training function, and performing post-training calculations, which are saved and returned in a dictionary

    def initialize_net(input_feat,output_dim,list_units,list_act,start_from_fc,params_dict,check_after=False):
    # Function initializing the re-forecaster with a pretrained network if warm start applied
        init_net = ct.NeuralNet(input_feat,output_dim,list_units, list_act)

        if start_from_fc:
            if check_after or list_units == []:
                idx_fc = params_dict['feat_cols'].index('y_hat')
                with torch.no_grad():
                    for i in range(init_net.final_layer.weight.shape[0]):
                        init_net.final_layer.bias[i] = 0
                        for j in range(init_net.final_layer.weight.shape[1]):
                            if j == params_dict['n_diff_features'] * i + idx_fc:
                                init_net.final_layer.weight[i, j] = 1
                            else:
                                init_net.final_layer.weight[i, j] = 0

            else:

                loc = "../data/pretrained_fc/model_softplus_wd_nlfr_genfc_yhat_scaled_pos/"

                weights_layer_1 = torch.tensor(np.load(loc + 'weights_layer_1.npz'),dtype=torch.float32)
                biases_layer_1 = torch.tensor(np.load(loc + 'biases_layer_1.npz'),dtype=torch.float32)
                weights_layer_2 = torch.tensor(np.load(loc + 'weights_layer_2.npz'),dtype=torch.float32)
                biases_layer_2 = torch.tensor(np.load(loc + 'biases_layer_2.npz'),dtype=torch.float32)

                with torch.no_grad():
                    init_net.hidden_layers[0].weight.copy_(weights_layer_1)
                    init_net.hidden_layers[0].bias.copy_(biases_layer_1)
                    init_net.final_layer.weight.copy_(weights_layer_2)
                    init_net.final_layer.bias.copy_(biases_layer_2)

        return init_net

    #Load data and relevant parameters
    list_tensors = input_dict['list_tensors']
    training_dict = input_dict['training_dict']
    params_dict = input_dict['params_dict']

    dev = training_dict['device']
    batch_size = training_dict['batch_size']
    model_type = training_dict['model_type']
    type_train_labels = training_dict['type_train_labels']

    print(f"Training config {input_dict['hp_config']}: gamma = {params_dict['gamma']}, reg = {training_dict['reg']}, lr = {training_dict['lr']}, batch size = {training_dict['batch_size']}")

    [train_feat,train_lab,val_feat,val_lab,test_feat,test_lab] = sf.set_tensors_to_device(list_tensors,dev)
    train_lab_training,val_lab_training,test_lab_training = sf.get_training_labels(train_lab,val_lab,test_lab)


    #Define dataloader
    random.seed(73)

    train_Dataset = sf.get_train_Dataset(model_type,type_train_labels,train_feat,train_lab_training)
    training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    training_dict['training_loader'] = training_loader
    training_dict['features'] = [train_feat,val_feat,test_feat]
    training_dict['labels'] = [train_lab_training,val_lab_training,test_lab_training]



    #Initializing the re-forecaster and loss function
    input_feat = train_feat.shape[1]
    output_dim = test_lab.shape[1]
    init_net = initialize_net(input_feat,output_dim,training_dict['list_units'],training_dict['list_act'],training_dict['start_from_fc'],params_dict)
    net_opti = ct.NeuralNetWithOpti(init_net,params_dict)
    profit_cal_fct = ct.Loss_profit()

    #Performance at start of training
    train_sched_0 = net_opti(train_feat)
    val_sched_0 = net_opti(val_feat)
    test_sched_0 = net_opti(test_feat)
    train_profit_0 = profit_cal_fct(train_sched_0,train_lab)
    val_profit_0 = profit_cal_fct(val_sched_0,val_lab)
    test_profit_0 = profit_cal_fct(test_sched_0,test_lab)

    print(f"Epoch 0 \t\t Train profit {train_profit_0} \t\t Validation profit: {val_profit_0} \t\t Test profit: {test_profit_0}")

    training_dict['net'] = net_opti
    training_dict['loss_fct'] = profit_cal_fct

    #Call training function
    tic=time.time()
    trained_net_opti,training_loss_best,valid_loss_best,test_loss_best,list_profit_evols, n_gradients_zero = train_nn(training_dict)
    train_time = time.time()-tic


    #Calculate perfect foresight performance
    optiLayer_RN = training_dict['optiLayer_RN']
    profit_train_pf = -profit_cal_fct(optiLayer_RN(train_lab),train_lab).item()
    profit_val_pf = -profit_cal_fct(optiLayer_RN(val_lab),val_lab).item()
    profit_test_pf = -profit_cal_fct(optiLayer_RN(test_lab),test_lab).item()

    #Determine best net and calculate train/validation/test performance
    #We differentiate profit_train and profit_RN_train. The former calculates the profit obtained by deploying the modified optimization program, the latter considers the original optimization program
    idx = list_profit_evols[3].index(max(list_profit_evols[3]))

    profit_train = list_profit_evols[0][idx]
    profit_val = list_profit_evols[2][idx]
    profit_test = list_profit_evols[4][idx]

    profit_RN_train = list_profit_evols[1][idx]
    profit_RN_val = list_profit_evols[3][idx]
    profit_RN_test = list_profit_evols[5][idx]

    regret_RN_train = profit_train_pf - profit_RN_train
    regret_RN_val = profit_val_pf - profit_RN_val
    regret_RN_test = profit_test_pf - profit_RN_test


    #Check profit when using pretrained price forecasts
    price_fc = initialize_net(input_feat,output_dim,[],[],True,params_dict,check_after=False)
    profit_RN_train_fc = -profit_cal_fct(optiLayer_RN(price_fc(train_feat)),train_lab).item()
    profit_RN_val_fc = -profit_cal_fct(optiLayer_RN(price_fc(val_feat)), val_lab).item()
    profit_RN_test_fc = -profit_cal_fct(optiLayer_RN(price_fc(test_feat)),test_lab).item()


    #Aggregate all relevant information to be stored in dictionary to be stored
    output_dict = {
        'profit_evol_train_RA': list_profit_evols[0],
        'profit_evol_train_RN': list_profit_evols[1],
        'profit_evol_val_RA': list_profit_evols[2],
        'profit_evol_val_RN': list_profit_evols[3],
        'profit_evol_test_RA': list_profit_evols[4],
        'profit_evol_test_RN': list_profit_evols[5],
        'regret_evol_train': [profit_train_pf - x for x in list_profit_evols[1]],
        'regret_evol_val': [profit_val_pf - x for x in list_profit_evols[3]],
        'regret_evol_test': [profit_test_pf - x for x in list_profit_evols[5]],
        'n_gradients_zero': n_gradients_zero,
        'a_config': input_dict['hp_config'],
        'trained_net_opti': trained_net_opti,
        'a_profit_train':profit_train,
        'a_profit_val':profit_val,
        'a_profit_test': profit_test,
        'a_profit_RN_train': profit_RN_train,
        'a_profit_RN_val': profit_RN_val,
        'a_profit_RN_test': profit_RN_test,
        'a_profit_RN_train_fc': profit_RN_train_fc,
        'a_profit_RN_val_fc': profit_RN_val_fc,
        'a_profit_RN_test_fc': profit_RN_test_fc,
        'ab_profit_PF_train': profit_train_pf,
        'ab_profit_PF_val': profit_val_pf,
        'ab_profit_PF_test': profit_test_pf,
        'b_train_time': train_time,
        'hp_reg': training_dict['reg'],
        'hp_gamma': params_dict['gamma'],
        'hp_lr': training_dict['lr'],
        'hp_bs': training_dict['batch_size'],
        'list_units': training_dict['list_units'],
        'list_act': training_dict['list_act']
    }



    return output_dict

def train_nn(dict):
    #Function that trains the neural network given the specified train data. It returns the best net, the train/val/test loss of the best net, train/val/test loss evolution through the training procedure, as well as the evolution of zeros in the gradient

    def clone_model(model):
        #Function ensuring that when new optimum is found, that specific forecaster is not overwritten by subsequent gradient updates
        price_gen = type(model.price_generator)(model.price_generator.input_feat,model.price_generator.output_dim,model.price_generator.list_units,model.price_generator.list_act)
        price_gen.load_state_dict(model.price_generator.state_dict())
        try:
            clone=type(model)(price_gen,model.params_dict)
        except Exception as e:
            print("cloning failed")
            test = sf.OptiLayer(model.params_dict)
            clone=model
            #clone.load_state_dict(model.state_dict())

        return clone

    #Extract relevant information from input dict
    net = dict['net']
    training_loader = dict['training_loader']
    opti_layer_RN = dict['optiLayer_RN']
    model_type = dict['model_type']
    if 'loss_fct' in dict:
        loss_fct = dict['loss_fct']
    else:
        loss_fct = torch.nn.MSELoss()

    if 'features' in dict:
        features = dict['features']
    else:
        features = ['NA','NA','NA']


    if 'labels' in dict:
        labels = dict['labels']
    else:
        labels = ['NA','NA', 'NA']

    if 'epochs' in dict:
        epochs = dict['epochs']
    else:
        epochs = 30

    if 'lr' in dict:
        lr = dict['lr']
    else:
        lr = 0.001

    if 'patience' in dict:
        patience = dict['patience']
    else:
        patience = 10

    if 'rd_seed' in dict:
        rd_seed = dict['rd_seed']
    else:
        rd_seed = 73

    if 'reg' in dict:
        reg = dict['reg']
    else:
        reg = 0




    ### INITIALIZATION ###

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    valid_loss_best = np.inf
    test_loss_best = np.inf
    train_loss_best = 0

    X_train = features[0]
    X_val = features[1]
    X_test = features[2]
    Y_train = labels[0]
    Y_val = labels[1]
    Y_test = labels[2]

    loss_evolution = np.zeros(epochs)

    epochs_since_improvement = 0

    torch.manual_seed(rd_seed)

    #Keep track of profit over time
    profit_train_RA = [-loss_fct(net(X_train),Y_train).item()]
    profit_train_RN = [-loss_fct(opti_layer_RN(net.price_generator(X_train)),Y_train).item()]
    profit_val_RA = [-loss_fct(net(X_val), Y_val).item()]
    profit_val_RN = [-loss_fct(opti_layer_RN(net.price_generator(X_val)), Y_val).item()]
    profit_test_RA = [-loss_fct(net(X_test), Y_test).item()]
    profit_test_RN = [-loss_fct(opti_layer_RN(net.price_generator(X_test)), Y_test).item()]
    n_gradients_zero = [0]

    for e in range(epochs):

        if epochs_since_improvement >= patience:
            break

        ### TRAINING PHASE ###
        train_loss_RA = 0.0
        train_start = time.time()

        #We keep track of the gradients in the final layer
        final_layer_grad = np.zeros_like(net.price_generator.final_layer.weight.detach().numpy())

        for i, data in enumerate(training_loader):

            if len(data) == 3:
                inputs, labels_price, labels_schedule = data
                # clear gradients
                optimizer.zero_grad()
                # Forward pass
                price_fc,sched_fc = net(inputs)

                loss = loss_fct(sched_fc, price_fc, labels_schedule, labels_price)
            else:
                inputs,labels = data
                optimizer.zero_grad()
                fc = net(inputs)
                loss = loss_fct(fc,labels) + reg * net.regularization()

            loss.backward()

            final_layer_grad += net.price_generator.final_layer.weight.grad.detach().numpy()

            # Update weights
            optimizer.step()
            train_loss_RA += loss.item()

        #Calculate of the amount of zeros in gradient
        th = 1e-5
        zeros_final = (np.abs(final_layer_grad) < th).sum()

        loss_evolution[e] = train_loss_RA


        ### VALIDATION PHASE ###
        if X_val == 'NA':
            pass
        else:
            #We differentiate loss_RA, which is the loss from the decisions based on the modified problem (which behaves like a risk-averse problem) and loss_RN, which is the loss from the decisions taken by the original, unmodified optimization program
            train_loss_RA = loss_fct(net(X_train),Y_train).item()
            train_loss_RN = loss_fct(opti_layer_RN(net.price_generator(X_train)),Y_train).item()
            val_loss_RA = loss_fct(net(X_val), Y_val).item()
            val_loss_RN = loss_fct(opti_layer_RN(net.price_generator(X_val)),Y_val).item()
            test_loss_RA = loss_fct(net(X_test), Y_val).item()
            test_loss_RN = loss_fct(opti_layer_RN(net.price_generator(X_test)),Y_test).item()


            print(f'Epoch {e + 1} \t\t Training Loss RA: {train_loss_RA} \t\t Validation Loss RA: {val_loss_RA} \t\t Test Loss RA: {test_loss_RA}')
            print(f'Epoch {e + 1} \t\t Training Loss RN: {train_loss_RN} \t\t Validation Loss RN: {val_loss_RN} \t\t Test Loss RN: {test_loss_RN}')

            profit_train_RA.append(-train_loss_RA)
            profit_train_RN.append(-train_loss_RN)
            profit_val_RA.append(-val_loss_RA)
            profit_val_RN.append(-val_loss_RN)
            profit_test_RA.append(-test_loss_RA)
            profit_test_RN.append(-test_loss_RN)
            n_gradients_zero.append(zeros_final)

            if valid_loss_best > val_loss_RN:
                print(f'Validation Loss RN Decreased({valid_loss_best:.6f}--->{val_loss_RN:.6f}) \t Saving The Model')
                valid_loss_best = val_loss_RN
                train_loss_best = train_loss_RN
                test_loss_best = test_loss_RN
                epochs_since_improvement = 0
                #best_net = copy.deepcopy(net)
                best_net = clone_model(net)
            else:
                epochs_since_improvement += 1


    list_profit_evols = [profit_train_RA,profit_train_RN,profit_val_RA,profit_val_RN,profit_test_RA,profit_test_RN]


    return best_net, train_loss_best, valid_loss_best, test_loss_best, list_profit_evols, n_gradients_zero

