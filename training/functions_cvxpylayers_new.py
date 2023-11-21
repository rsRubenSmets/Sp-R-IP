import torch
import functions_support_new as sf
import torch_classes_new as ct
import numpy as np
import time
import random
import sys
import torch_classes_new
import copy

def hp_tuning_cvxpylayers_2(input_dict):
    #Overarching function of the training process, unpacking all the relevant data of the input dict, calling the training function, and performing post-training calculations, which are saved and returned in a dictionary

    def initialize_net(input_feat,output_dim,list_units,list_act,start_from_fc,params_dict,check_after=False):
    # Function initializing the re-forecaster with a pretrained network if warm start applied
        torch.manual_seed(71)

        init_net = ct.NeuralNet(input_feat=input_feat,output_dim=output_dim,list_units=list_units, list_act=list_act)

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

    train_Dataset = torch_classes_new.Dataset_Lists(train_feat, train_lab)
    training_loader = torch.utils.data.DataLoader(train_Dataset,batch_size=batch_size,shuffle=True)

    # train_Dataset = sf.get_train_Dataset(model_type,type_train_labels,train_feat,train_lab_training)
    # training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    training_dict['training_loader'] = training_loader
    training_dict['features'] = [train_feat,val_feat,test_feat]
    training_dict['labels'] = [train_lab_training,val_lab_training,test_lab_training]

    loss_fct_eval_strs = ['profit', 'mse_sched', 'mse_sched_weighted','mse_price']
    training_dict['loss_fct_eval_strs'] = loss_fct_eval_strs

    #Initializing the re-forecaster and loss function
    input_feat = train_feat[0].shape[1]
    output_dim = test_lab[0].shape[1]
    init_net = initialize_net(input_feat,output_dim,training_dict['list_units'],training_dict['list_act'],training_dict['start_from_fc'],params_dict)
    net_opti = ct.NeuralNetWithOpti(init_net,params_dict)
    profit_cal_fct = ct.Loss_smoothing('profit')

    #Performance at start of training
    train_sched_0 = net_opti(train_feat)
    val_sched_0 = net_opti(val_feat)
    test_sched_0 = net_opti(test_feat)
    train_profit_0 = profit_cal_fct([train_sched_0],train_lab)
    val_profit_0 = profit_cal_fct([val_sched_0],val_lab)
    test_profit_0 = profit_cal_fct([test_sched_0],test_lab)

    print(f"Epoch 0 \t\t Train profit {train_profit_0} \t\t Validation profit: {val_profit_0} \t\t Test profit: {test_profit_0}")

    # if training_dict['framework'] == 'ID':
    #     training_dict['net'] = net_opti
    # elif training_dict['framework'] == 'SM':
    #     training_dict['net'] = init_net
    #     #training_dict['gc'] = ct.Grad_Calculator(params_dict, "quadratic", params_dict['gamma'])
    #     params_dict['smoothing'] = training_dict['smoothing']
    #     training_dict['sc'] = ct.Schedule_Calculator(params_dict)

    params_dict['smoothing'] = training_dict['smoothing']

    net = ct.NeuralNetWithOpti_new(init_net,params_dict,training_dict['framework'])
    training_dict['net'] = net

    training_dict['loss_fct'] = ct.Loss_smoothing(training_dict['loss_fct_str'])

    #Call training function
    tic=time.time()
    # if training_dict['framework'] == "ID":
    #     trained_net_opti,training_loss_best,valid_loss_best,test_loss_best,list_profit_evols, n_gradients_zero = train_nn(training_dict)
    # elif training_dict['framework'] == "SM":
    #     trained_net_opti,training_loss_best,valid_loss_best,test_loss_best,loss_evols = train_nn_smoothing(training_dict)

    trained_net_opti,training_loss_best,valid_loss_best,test_loss_best,loss_evols,n_epochs, times = train_nn_smoothing(training_dict)

    train_time = time.time()-tic
    time_forward = times[0]
    time_backward = times[1]
    time_val = times[2]


    #Calculate perfect foresight performance
    optiLayer_RN = training_dict['optiLayer_RN']
    if type(train_lab) is list:
        profit_train_pf = -profit_cal_fct([optiLayer_RN(train_lab[0])],train_lab).item()
    else:
        profit_train_pf = -profit_cal_fct(optiLayer_RN(train_lab),train_lab).item()

    profit_train_pf = -profit_cal_fct([optiLayer_RN(train_lab[0])],train_lab).item()
    profit_val_pf = -profit_cal_fct([optiLayer_RN(val_lab[0])],val_lab).item()
    profit_test_pf = -profit_cal_fct([optiLayer_RN(test_lab[0])],test_lab).item()

    #Determine best net and calculate train/validation/test performance
    #We differentiate profit_train and profit_RN_train. The former calculates the profit obtained by deploying the modified optimization program, the latter considers the original optimization program
    idx = loss_evols['profit'][1].index(min(loss_evols['profit'][1]))

    profit_train = loss_evols['profit'][0][idx]
    profit_val = loss_evols['profit'][1][idx]
    profit_test = loss_evols['profit'][2][idx]

    # profit_RN_train = list_profit_evols[1][idx]
    # profit_RN_val = list_profit_evols[3][idx]
    # profit_RN_test = list_profit_evols[5][idx]
    #
    # regret_RN_train = profit_train_pf - profit_RN_train
    # regret_RN_val = profit_val_pf - profit_RN_val
    # regret_RN_test = profit_test_pf - profit_RN_test


    #Check profit when using pretrained price forecasts
    price_fc = initialize_net(input_feat,output_dim,[],[],True,params_dict,check_after=False)

    profit_train_fc = -profit_cal_fct([optiLayer_RN(price_fc(train_feat))],train_lab).item()
    profit_val_fc = -profit_cal_fct([optiLayer_RN(price_fc(val_feat))],val_lab).item()
    profit_test_fc = -profit_cal_fct([optiLayer_RN(price_fc(test_feat))],test_lab).item()


    #Aggregate all relevant information to be stored in dictionary to be stored
    output_dict = {
        'a_config': input_dict['hp_config'],
        'trained_net_opti': trained_net_opti,
        'a_profit_train':profit_train,
        'a_profit_val':profit_val,
        'a_profit_test': profit_test,
        'a_profit_train_fc': profit_train_fc,
        'a_profit_val_fc': profit_val_fc,
        'a_profit_test_fc': profit_test_fc,
        'ab_profit_PF_train': profit_train_pf,
        'ab_profit_PF_val': profit_val_pf,
        'ab_profit_PF_test': profit_test_pf,
        'b_train_time_per_epoch': train_time/n_epochs,
        'b_train_time_forward': time_forward/n_epochs,
        'b_train_time_backward': time_backward / n_epochs,
        'b_train_time_val': time_val / n_epochs,
        'hp_reg': training_dict['reg'],
        'hp_gamma': params_dict['gamma'],
        'hp_lr': training_dict['lr'],
        'hp_bs': training_dict['batch_size'],
        'hp_ws': training_dict['start_from_fc'],
        'smoothing': training_dict['smoothing'],
        'loss': training_dict['loss_fct_str'],
        'list_units': training_dict['list_units'],
        'list_act': training_dict['list_act'],
        'framework': training_dict['framework']
    }

    for loss_str in loss_fct_eval_strs:
        for (i,set) in enumerate(['train', 'val', 'test']):
            output_dict[f"loss_{loss_str}_{set}"] = loss_evols[loss_str][i]



    return output_dict


def train_nn_smoothing(dict):
    #Function that trains the neural network given the specified train data. It returns the best net, the train/val/test loss of the best net, train/val/test loss evolution through the training procedure, as well as the evolution of zeros in the gradient

    def clone_model(model):
        #Function ensuring that when new optimum is found, that specific forecaster is not overwritten by subsequent gradient updates

        # clone = type(model)(model.input_feat,model.output_dim,model.list_units,model.list_act)
        # clone.load_state_dict(model.state_dict())

        #return clone

        price_gen = model.price_generator
        clone_price_gen = type(price_gen)(price_gen.input_feat,price_gen.output_dim,price_gen.list_units,price_gen.list_act)
        clone_price_gen.load_state_dict(price_gen.state_dict())

        return type(model)(clone_price_gen,model.params_dict,model.framework)

    def append_losses(net,loss_fcts_eval,loss_evolution):
        price_fc_train = net.price_generator(X_train)
        price_fc_val = net.price_generator(X_val)
        price_fc_test = net.price_generator(X_test)
        # _, sched_out_train, _ = sc(price_fc_train)
        # _, sched_out_val, _ = sc(price_fc_val)
        # _, sched_out_test, _ = sc(price_fc_test)
        sched_out_train = net.calc_sched_linear(prices=price_fc_train)
        sched_out_val = net.calc_sched_linear(prices=price_fc_val)
        sched_out_test = net.calc_sched_linear(prices=price_fc_test)


        for loss_str in loss_fcts_eval.keys():
            loss_evolution[loss_str][0].append(loss_fcts_eval[loss_str]([sched_out_train,price_fc_train], Y_train).item())
            loss_evolution[loss_str][1].append(loss_fcts_eval[loss_str]([sched_out_val,price_fc_val], Y_val).item())
            loss_evolution[loss_str][2].append(loss_fcts_eval[loss_str]([sched_out_test,price_fc_test], Y_test).item())

        return loss_evolution

    #Extract relevant information from input dict
    net = dict['net']
    training_loader = dict['training_loader']
    opti_layer_RN = dict['optiLayer_RN']

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

    if 'loss_fct_eval_strs' in dict:
        loss_fcts_eval_strs = dict['loss_fct_eval_strs']
    else:
        loss_fcts_eval_strs = ['profit']

    if 'loss_fct_decision' in dict:
        loss_fct_decision = dict['loss_fct_decision']
    else:
        loss_fct_decision = 'profit'


    ### INITIALIZATION ###
    optimizer = torch.optim.Adam(net.price_generator.parameters(), lr=lr)
    valid_loss_best = np.inf
    test_loss_best = np.inf
    train_loss_best = 0

    X_train = features[0]
    Y_train = labels[0]
    X_val = features[1]
    Y_val = labels[1]
    X_test = features[2]
    Y_test = labels[2]

    #Initialize loss functions for evaluating performance
    loss_fcts_eval = {}
    loss_evolution = {}
    for loss_str in loss_fcts_eval_strs:
        loss_evolution[loss_str] = [[],[],[]]
        loss_fcts_eval[loss_str] = ct.Loss_smoothing(loss_str)

    epochs_since_improvement = 0

    torch.manual_seed(rd_seed)

    loss_evolution = append_losses(net,loss_fcts_eval,loss_evolution)

    time_forward = 0
    time_backward = 0
    time_val = 0


    print(f"Starting training of model with smoothing {net.sc.sm} using loss fct {loss_fct.obj}")

    ### TRAINING LOOP ###

    for e in range(epochs):

        if epochs_since_improvement >= patience:
            break

        ### TRAINING PHASE ###
        train_start = time.time()

        #We keep track of the gradients in the final layer
        final_layer_grad = np.zeros_like(net.price_generator.final_layer.weight.detach().numpy())

        for i, data in enumerate(training_loader):

            tic_start_forward = time.time()

            inputs = data[0]

            optimizer.zero_grad()
            price_fc = net.price_generator(inputs)
            sched_sm = net(inputs)


            tic_end_forward = time.time()

            #price_fc_with_custom_grad = ct.CustomGradFunction.apply(price_fc, labels_price, gc)
            loss = loss_fct([sched_sm,price_fc],data[1])
            loss.backward()

            # final_layer_grad += net.final_layer.weight.grad.detach().numpy()
            # mu_sm = gc.retrieve_mu(price_fc).detach().numpy()
            # sched_sm = gc.forward_pass(price_fc)
            # grads_prices = gc(price_fc,labels_price).detach().numpy()
            # price_fc = price_fc.detach().numpy()
            # price_act = labels_price.detach().numpy()

            # Update weights
            optimizer.step()

            tic_end_backward = time.time()

            print(f"forward pass: {tic_end_forward-tic_start_forward}; backward pass: {tic_end_backward-tic_end_forward}")

        th = 1e-5
        zeros_final = (np.abs(final_layer_grad) < th).sum()

        ### VALIDATION PHASE ###
        if X_val == 'NA':
            pass
        else:
            #Update all loss functions
            loss_evolution = append_losses(net,loss_fcts_eval,loss_evolution)
            train_loss = loss_evolution[loss_fct_decision][0][-1]
            val_loss = loss_evolution[loss_fct_decision][1][-1]
            test_loss = loss_evolution[loss_fct_decision][2][-1]

            tic_end_validation = time.time()


            #print(f'Epoch {e + 1} \t\t Training Loss RA: {train_loss_RA} \t\t Validation Loss RA: {val_loss_RA} \t\t Test Loss RA: {test_loss_RA}')
            print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss} \t\t Test Loss: {test_loss}')

            if valid_loss_best > val_loss:
                print(f'Validation Loss RN Decreased({valid_loss_best:.6f}--->{val_loss:.6f}) \t Saving The Model')
                valid_loss_best = val_loss
                train_loss_best = train_loss
                test_loss_best = test_loss
                epochs_since_improvement = 0
                best_net = clone_model(net)
            else:
                epochs_since_improvement += 1

            time_forward += tic_end_forward-tic_start_forward
            time_backward += tic_end_backward - tic_end_forward
            time_val += tic_end_validation - tic_end_backward


    return best_net, train_loss_best, valid_loss_best, test_loss_best, loss_evolution, e+1, [time_forward,time_backward,time_val]



##### OLD FUNCTIONS #####

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
    Y_train = labels[0]
    X_val = features[1]
    Y_val = labels[1]
    X_test = features[2]
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
