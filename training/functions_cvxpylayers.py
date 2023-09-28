import torch
import functions_support as sf
import classes_torch as ct
import numpy as np
import time
import random
import copy




def train_nn(dict):

    def clone_model(model):


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
    #best_net = copy.deepcopy(net)
    #best_net = net

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

        final_layer_grad = np.zeros_like(net.price_generator.final_layer.weight.detach().numpy())
        if net.price_generator.hidden_layers != []:
            first_layer_grad = np.zeros_like(net.price_generator.hidden_layers[0].weight.detach().numpy())

        for i, data in enumerate(training_loader):
            if model_type == 'edRNN':
                inputs_e, inputs_d, labels = data
                optimizer.zero_grad()
                price_fc = net(inputs_e, inputs_d)

                loss = loss_fct(y_pred=price_fc, y_true=labels) #+lambda*(model.regularization())


            else:
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

            # Calculate loss
            #loss = loss_fct(y_pred=price_fc, y_true=labels)
            # Calculate gradients
            loss.backward()

            final_layer_grad += net.price_generator.final_layer.weight.grad.detach().numpy()
            if net.price_generator.hidden_layers != []:
                first_layer_grad += net.price_generator.hidden_layers[0].weight.grad.detach().numpy()



            # Update weights
            optimizer.step()
            """
            if math.isnan(net.lstm_d.all_weights[0][0].to('cpu').detach().numpy()[0,0]):
                x=1
            """
            train_loss_RA += loss.item()


        th = 1e-5
        zeros_final = (np.abs(final_layer_grad) < th).sum()
        if net.price_generator.hidden_layers != []:
            zeros_first = (np.abs(first_layer_grad) < th).sum()

        #print(f"Zeros in first layer: {zeros_first}; Zeros in final layer: {zeros_final}")

        train_time = time.time() - train_start

        loss_evolution[e] = train_loss_RA
        #print(f'Epoch {e + 1} \t\t Training Loss: {train_loss_RA} \t\t Train time: {train_time}'  )


        ### VALIDATION PHASE ###
        if X_val == 'NA':
            pass
        else:
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

def hp_tuning_cvxpylayers_2(input_dict):

    def initialize_net(input_feat,output_dim,list_units,list_act,start_from_fc,params_dict,check_after=False):

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

    random.seed(73)

    train_Dataset = sf.get_train_Dataset(model_type,type_train_labels,train_feat,train_lab_training)
    training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    training_dict['training_loader'] = training_loader
    training_dict['features'] = [train_feat,val_feat,test_feat]
    training_dict['labels'] = [train_lab_training,val_lab_training,test_lab_training]




    input_feat = train_feat.shape[1]
    output_dim = test_lab.shape[1]
    init_net = initialize_net(input_feat,output_dim,training_dict['list_units'],training_dict['list_act'],training_dict['start_from_fc'],params_dict)
    #init_net = initialize_net(input_feat,output_dim,training_dict['list_units'],training_dict['list_act'],training_dict['start_from_fc'],params_dict,True)


    if input_dict['hp_config'] >1:
        x=1 #for test purpose



    #pretrained_fc, train_loss_best, valid_loss_best, test_loss_best = train_nn(training_dict)


    net_opti = ct.NeuralNetWithOpti(init_net,params_dict)

    profit_cal_fct = ct.Loss_profit()

    train_sched_0 = net_opti(train_feat)
    val_sched_0 = net_opti(val_feat)
    test_sched_0 = net_opti(test_feat)
    train_profit_0 = profit_cal_fct(train_sched_0,train_lab)
    val_profit_0 = profit_cal_fct(val_sched_0,val_lab)
    test_profit_0 = profit_cal_fct(test_sched_0,test_lab)

    # p = net_opti.price_generator(train_feat).detach().numpy()
    # s = net_opti(train_feat).detach().numpy()
    #
    # prob, [price], [net_discharge,e_d,e_c,soc,degr_cost] = sf.opti_problem(params_dict)
    # price.value = p[0,:]
    #
    # prob.solve()
    #
    # gamma = 3
    # import cvxpy as cp
    # obj = -price.value@net_discharge.value + gamma * cp.norm(net_discharge.value,2)


    print(f"Epoch 0 \t\t Train profit {train_profit_0} \t\t Validation profit: {val_profit_0} \t\t Test profit: {test_profit_0}")


    training_dict['net'] = net_opti
    training_dict['loss_fct'] = profit_cal_fct

    #trained_net_opti = net_opti
    tic=time.time()
    trained_net_opti,training_loss_best,valid_loss_best,test_loss_best,list_profit_evols, n_gradients_zero = train_nn(training_dict)
    train_time = time.time()-tic


    #direct calculation of profit :
    # profit_train = -profit_cal_fct(trained_net_opti(train_feat),train_lab).item()
    # profit_val = -profit_cal_fct(trained_net_opti(val_feat),val_lab).item()
    # profit_test = -profit_cal_fct(trained_net_opti(test_feat),test_lab).item()
    #
    # #Implementation with intermediate price in RN optimization program
    optiLayer_RN = training_dict['optiLayer_RN']
    # profit_RN_train = -profit_cal_fct(optiLayer_RN(trained_net_opti.get_intermediate_price_prediction(train_feat)),train_lab).item()
    # profit_RN_val = -profit_cal_fct(optiLayer_RN(trained_net_opti.get_intermediate_price_prediction(val_feat)),val_lab).item()
    # profit_RN_test = -profit_cal_fct(optiLayer_RN(trained_net_opti.get_intermediate_price_prediction(test_feat)),test_lab).item()

    profit_train_pf = -profit_cal_fct(optiLayer_RN(train_lab),train_lab).item()
    profit_val_pf = -profit_cal_fct(optiLayer_RN(val_lab),val_lab).item()
    profit_test_pf = -profit_cal_fct(optiLayer_RN(test_lab),test_lab).item()




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













#### OLD FUNCTIONS

# def hp_tuning_cvxpylayers(list_tensors, training_dict, params_dict):
#     dev = training_dict['device']
#     decomp = training_dict['decomp']
#     batch_size = training_dict['batch_size']
#     model_type = training_dict['model_type']
#     type_train_labels = training_dict['type_train_labels']
#
#     [train_feat, train_lab, val_feat, val_lab, test_feat, test_lab] = sf.set_tensors_to_device(list_tensors, dev)
#
#     train_lab_training, val_lab_training, test_lab_training = sf.get_training_labels(train_lab, val_lab, test_lab,
#                                                                                      decomp)
#
#     train_Dataset = sf.get_train_Dataset(model_type, type_train_labels, train_feat, train_lab_training)
#     training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)
#
#     training_dict['training_loader'] = training_loader
#     training_dict['val_test_feat'] = [val_feat, test_feat]
#     training_dict['val_test_lab_training'] = [val_lab_training, test_lab_training]
#
#     pretrained_fc = training_dict['net']
#     # pretrained_fc, train_loss_best, valid_loss_best, test_loss_best = train_nn(training_dict)
#
#     net_opti = sf.NeuralNetWithOpti(pretrained_fc, params_dict)
#
#     profit_cal_fct = sf.Loss_profit()
#
#     train_sched_0 = net_opti(train_feat)
#     profit_0 = profit_cal_fct(train_sched_0, train_lab)
#
#     print(f"Epoch 0: profit {profit_0}")
#
#     training_dict['net'] = net_opti
#     training_dict['loss_fct'] = profit_cal_fct
#     training_dict['epochs'] = 100
#     training_dict['patience'] = 10
#
#     # trained_net_opti = net_opti
#     trained_net_opti, training_loss_best, valid_loss_best, test_loss_best = train_nn(training_dict)
#
#
#
#     return pretrained_fc, trained_net_opti