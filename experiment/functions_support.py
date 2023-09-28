##### LOSS FUNCTIONS AND NN ARCHITECTURES #####

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable,Function
import pywt
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
import h5py
import csv
from pyomo.environ import *
import pyomo.environ as pyo
from pyomo.opt import SolverFactory



##### LOSS CLASSES #####
class Generalized_loss_eff_old(nn.Module):

    def __init__(self, range_p, weights, range_n=0, range_k=0):
        super(Generalized_loss_eff_old, self).__init__()
        self.range_p = range_p
        self.range_n = range_n
        self.range_k = range_k
        self.weights = weights

        # assert len(weights_norm) == max_p, f"Expected {max_p} normal weights, got: {len(weights_norm)}"

    def forward(self, y_true, y_pred):

        loss = 0
        [n_examples, len_pred] = y_true.size()
        abs_diff = torch.zeros(n_examples, len_pred, len_pred)

        abs_diff[:, :, 0] = torch.abs(y_true - y_pred)

        for n in range(len_pred - 1):
            abs_diff[:, 0:len_pred - n - 1, n + 1] = torch.abs((y_true[:, 0:len_pred - n - 1] - y_true[:, n + 1:]) - (
                        y_pred[:, 0:len_pred - n - 1] - y_pred[:, n + 1:]))

        for p in range(len(self.range_p)):
            loss += torch.sum(torch.mul(torch.pow(abs_diff, self.range_p[p]), torch.from_numpy(self.weights[p, :, :])))

        return loss / len_pred / n_examples


class Generalized_loss_eff(nn.Module):

    def __init__(self, range_p, weights):
        super(Generalized_loss_eff, self).__init__()
        self.range_p = range_p
        self.weights = torch.from_numpy(weights)

        # assert len(weights_norm) == max_p, f"Expected {max_p} normal weights, got: {len(weights_norm)}"

    def forward(self, y_true, y_pred):

        loss = 0
        [n_examples, len_pred] = y_true.size()
        abs_diff = torch.zeros(n_examples, len_pred, len_pred)

        abs_diff[:, :, 0] = torch.abs(y_true - y_pred)

        for n in range(len_pred - 1):
            abs_diff[:, 0:len_pred - n - 1, n + 1] = torch.abs((y_true[:, 0:len_pred - n - 1] - y_true[:, n + 1:]) - (
                        y_pred[:, 0:len_pred - n - 1] - y_pred[:, n + 1:]))

        for p in range(len(self.range_p)):
            loss += torch.pow(torch.sum(torch.pow(torch.mul(self.weights[p, :], abs_diff), self.range_p[p])),
                              1 / self.range_p[p])

        return loss / len_pred / n_examples


class Loss_levelized_p(nn.Module):

    def __init__(self, range_p, weights, batch_size):
        super(Loss_levelized_p, self).__init__()
        self.range_p = range_p
        self.weights = self.scaled_weights(weights, batch_size)

        # assert len(weights_norm) == max_p, f"Expected {max_p} normal weights, got: {len(weights_norm)}"

    def forward(self, y_true, y_pred):

        loss = 0
        [n_examples, len_pred] = y_true.size()
        abs_diff = torch.zeros(n_examples, len_pred, len_pred)

        abs_diff[:, :, 0] = torch.abs(y_true - y_pred)

        for n in range(len_pred - 1):
            abs_diff[:, 0:len_pred - n - 1, n + 1] = torch.abs((y_true[:, 0:len_pred - n - 1] - y_true[:, n + 1:]) - (
                        y_pred[:, 0:len_pred - n - 1] - y_pred[:, n + 1:]))

        for p in range(len(self.range_p)):
            interm = torch.pow(torch.sum(torch.pow(torch.mul(self.weights[p, :], abs_diff), self.range_p[p])),
                               1 / self.range_p[p])
            loss += interm

        return loss / len_pred / n_examples

    def scaled_weights(self, weights, batch_size):
        p_max = self.range_p[-1]
        nonzero_weights = np.count_nonzero(weights[0, :, :]) * batch_size

        for p_ind in range(len(self.range_p)):
            p = self.range_p[p_ind]
            corr_exp = 1 / p_max - 1 / p
            corr_factor = 3 ** (p_max - p)

            weights[p_ind, :, :] = weights[p_ind, :, :] * corr_factor * nonzero_weights ** (corr_exp)

        return torch.from_numpy(weights)


class Generalized_loss_eff_decomp(nn.Module):

    def __init__(self, range_p, weights):
        super(Generalized_loss_eff_decomp, self).__init__()
        self.range_p = range_p
        self.weights = torch.from_numpy(weights)

        # assert len(weights_norm) == max_p, f"Expected {max_p} normal weights, got: {len(weights_norm)}"

    def forward(self, y_true, y_pred):
        loss = 0
        [n_modes, len_pred] = y_true.size()

        abs_diff = torch.abs(y_true - y_pred)

        for p in range(len(self.range_p)):
            loss += torch.pow(torch.sum(torch.pow(torch.mul(self.weights[p, :], abs_diff), self.range_p[p])),
                              1 / self.range_p[p])

        return loss / len_pred / n_modes


class Generalized_loss_eff_fg(nn.Module):

    def __init__(self, range_p, range_n, range_k, weights_matrix, fgv_level=np.inf):
        super(Generalized_loss_eff_fg, self).__init__()
        self.range_p = range_p
        self.weights = torch.from_numpy(weights_matrix)
        self.fgv_level = fgv_level

        # assert len(weights_norm) == max_p, f"Expected {max_p} normal weights, got: {len(weights_norm)}"

    def forward(self, y_true, y_pred, y_diff):

        loss = 0
        [n_examples, len_pred] = y_true.size()
        abs_diff = torch.zeros(n_examples, len_pred, len_pred)
        y_filter_fg = torch.zeros_like(abs_diff)
        filter = torch.where(torch.abs(y_diff) > self.fgv_level, 0, 1)

        abs_diff[:, :, 0] = torch.abs(y_true - y_pred)
        y_filter_fg[:, :, 0] = filter

        for n in range(len_pred - 1):
            abs_diff[:, 0:len_pred - n - 1, n + 1] = torch.abs((y_true[:, 0:len_pred - n - 1] - y_true[:, n + 1:]) - (
                        y_pred[:, 0:len_pred - n - 1] - y_pred[:, n + 1:]))
            y_filter_fg[:, 0:len_pred - n - 1, n + 1] = torch.mul(filter[:, 0:len_pred - n - 1], filter[:, n + 1:])

        for p in range(len(self.range_p)):
            loss += torch.sum(
                torch.mul(torch.pow(abs_diff, self.range_p[p]), torch.mul(self.weights[p, :, :], y_filter_fg)))

        return loss / len_pred / (n_examples * torch.mean(filter, dtype=torch.float))


class Loss_cross_profit(nn.Module):

    def __init__(self):
        super(Loss_cross_profit, self).__init__()

    def forward(self, schedule_net, price_net, schedule_opti, price_opti):
        loss = torch.sum(torch.square((torch.mul(schedule_net, price_opti) - torch.mul(schedule_opti, price_net))))

        return loss

class Loss_profit(nn.Module):

    def __init__(self):
        super(Loss_profit, self).__init__()

    def forward(self, output, prices):
        schedules = output
        profit = torch.sum(torch.mul(schedules, prices))

        return -profit


##### NN ARCHITECTURES #####

class LSTM_ED(torch.nn.Module):
    def __init__(self, input_size_e, seq_length_e, hidden_size, input_size_d, seq_length_d, dev):
        super(LSTM_ED, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size

        self.hidden_size = hidden_size  # hidden state
        self.seq_length_e = seq_length_e  # sequence length
        self.seq_length_d = seq_length_d  # sequence length

        self.num_layers_e = 1
        self.num_layers_d = 1

        self.dev = dev

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Decoder
        self.fc = torch.nn.Linear(hidden_size, 1)  # fully connected 1

    def forward(self, x_e, x_d, dev_type='NA'):
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))
        prices = torch.squeeze(self.fc(output_d))  # Final Output
        return prices

class LSTM_ED_decomp(torch.nn.Module):
    def __init__(self, input_size_e, seq_length_e, hidden_size, input_size_d, seq_length_d, n_decomp_modes, dev):
        super(LSTM_ED_decomp, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size

        self.hidden_size = hidden_size  # hidden state
        self.seq_length_e = seq_length_e  # sequence length
        self.seq_length_d = seq_length_d  # sequence length

        self.n_decomp_modes = n_decomp_modes

        self.num_layers_e = 1
        self.num_layers_d = 1

        self.dev = dev

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Decoder
        self.fc = torch.nn.Linear(hidden_size, n_decomp_modes)  # fully connected 1

    def forward(self, x_e, x_d, dev_type='NA'):
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))
        modes = torch.squeeze(self.fc(
            h_d))  # Only use final output for calculating the modes, as this is influenced by all previous instances

        return modes

class LSTM_ED_decomp_ext(torch.nn.Module):
    def __init__(self, input_size_e, seq_length_e, hidden_size, input_size_d, seq_length_d, n_decomp_modes, dev):
        super(LSTM_ED_decomp_ext, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size

        self.hidden_size = hidden_size  # hidden state
        self.seq_length_e = seq_length_e  # sequence length
        self.seq_length_d = seq_length_d  # sequence length

        self.n_decomp_modes = n_decomp_modes

        self.num_layers_e = 1
        self.num_layers_d = 1

        self.dev = dev

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d, hidden_size=hidden_size, num_layers=1,
                                    batch_first=True)  # Decoder
        self.fc_h = torch.nn.Linear(hidden_size, n_decomp_modes)  # fully connected 1
        self.fc_c = torch.nn.Linear(hidden_size, n_decomp_modes)

        self.activ_h = torch.nn.ReLU()
        self.activ_c = torch.nn.ReLU()

        self.final = torch.nn.Linear(n_decomp_modes * 2, n_decomp_modes)

    def forward(self, x_e, x_d, dev_type='NA'):
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))

        out_h = self.activ_h(self.fc_h(h_d))
        out_c = self.activ_c(self.fc_c(c_d))

        modes = torch.squeeze(self.final(torch.cat((out_h, out_c), dim=2)))

        # modes = torch.squeeze(self.fc(h_d)) #Only use final output for calculating the modes, as this is influenced by all previous instances

        return modes


class NeuralNetOld(torch.nn.Module):
    def __init__(self, n_units_1, n_units_2, act_1, act_2):
        super(NeuralNetOld, self).__init__()

        # Define layers
        self.layer_1 = torch.nn.Linear(29 * 10 + 4, n_units_1)
        self.layer_2 = torch.nn.Linear(n_units_1, n_units_2)
        self.final_layer = torch.nn.Linear(n_units_2, 10)
        # Define activations
        if act_1 == 'relu':
            self.act_fct_1 = F.relu
        elif act_1 == 'elu':
            self.act_fct_1 = F.elu
        elif act_1 == 'softplus':
            self.act_fct_1 = F.softplus
        if act_2 == 'relu':
            self.act_fct_2 = F.relu
        elif act_2 == 'elu':
            self.act_fct_2 = F.elu
        elif act_2 == 'softplus':
            self.act_fct_2 = F.softplus
        # self.act_fct_final = F.tanh

    def forward(self, x):
        pass_1 = self.act_fct_1(self.layer_1(x))
        pass_2 = self.act_fct_2(self.layer_2(pass_1))
        prices = self.final_layer(pass_2)
        return prices

class NeuralNet(torch.nn.Module):
    def __init__(self, input_feat, output_dim,list_units=[],list_act=[]):
        super(NeuralNet, self).__init__()

        self.input_feat = input_feat
        self.output_dim = output_dim
        self.list_units = list_units
        self.list_act = list_act

        dict_act_fcts = {
            'relu': F.relu,
            'elu': F.elu,
            'softplus': F.softplus
        }

        # Define layers

        self.hidden_layers = []
        self.act_fcts = []

        for i,units in enumerate(list_units):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(input_feat,units))
            else:
                self.hidden_layers.append(torch.nn.Linear(list_units[i-1],units))

            self.act_fcts.append(dict_act_fcts[list_act[i]])

        if len(list_units)>0:
            self.final_layer = torch.nn.Linear(list_units[-1], output_dim)
        else:
            self.final_layer = torch.nn.Linear(input_feat, output_dim)

    def regularization(self,pow=1):

        reg = 0
        for layer in self.hidden_layers:
            reg+=torch.sum(torch.pow(torch.abs(layer.weight),pow))

        reg+= torch.sum(torch.pow(torch.abs(self.final_layer.weight),pow))

        return reg

    def forward(self, x):

        for i,act in enumerate(self.act_fcts):

            x = act(self.hidden_layers[i](x))

        out = self.final_layer(x)

        return out

class OptiLayer(torch.nn.Module):
    def __init__(self, params_dict):
        super(OptiLayer, self).__init__()
        prob, params, vars = opti_problem(params_dict)
        self.layer = CvxpyLayer(prob, params, vars)

    def forward(self, x):
        #return self.layer(x, solver_args={'max_iters': 10000, 'solve_method': 'ECOS'})[0]  # 'eps': 1e-4,'mode':'dense'
        try:
            result = self.layer(x,solver_args={"solve_method": "ECOS"})[0] #"n_jobs_forward": 1
        except:
            print("solvererror occured")
            result = self.layer(x,solver_args={"solve_method": "SCS"})[0]


        return result

class NeuralNetWithOpti(torch.nn.Module):
    def __init__(self, price_gen,params_dict):
        super(NeuralNetWithOpti, self).__init__()
        self.price_generator = price_gen
        self.convex_opti_layer = OptiLayer(params_dict)
        self.params_dict = params_dict

    def forward(self, x):
        prices = self.price_generator(x)  # .is_leaf_(True)
        prices.retain_grad()
        schedule = self.convex_opti_layer(prices)
        return schedule

    def regularization(self):

        return self.price_generator.regularization()

    def get_intermediate_price_prediction(self,x):
        return self.price_generator(x)



def save_outcome(list_dict_outcome, store_code):
    # Function saving all information in a list of dicts coming out of the NN training procedure in specified location

    def convert_nn_arch(list_dict_outcome):
        n_hidden_layers = len(list_dict_outcome[0]['list_units'])

        for dict in list_dict_outcome:
            for hl in range(n_hidden_layers):
                dict[f'hp_hl{hl+1}_units'] = dict['list_units'][hl]
                dict[f'hp_hl{hl+1}_act'] = dict['list_act'][hl]

            del dict['list_units']
            del dict['list_act']

        return list_dict_outcome


    for dict_out in list_dict_outcome:
        path = f"{store_code}config_{dict_out['a_config']}.pt"
        torch.save(dict_out['trained_net_opti'], path)
        del dict_out['trained_net_opti']

        path = f"{store_code}config_{dict_out['a_config']}_profitEvol.npz"
        path_h5py = f"{store_code}config_{dict_out['a_config']}_train_evols.h5"
        save_arrays = ['profit_evol_train_RA','profit_evol_train_RN','profit_evol_val_RA','profit_evol_val_RN','profit_evol_test_RA','profit_evol_test_RN','n_gradients_zero','regret_evol_train', 'regret_evol_val', 'regret_evol_test']

        with h5py.File(path_h5py, 'w') as f:
            for key, value in {k:dict_out[k] for k in save_arrays}.items():
                f.create_dataset(key, data=value)

        for evol in save_arrays:
            del dict_out[evol]


    list_dict_outcome = convert_nn_arch(list_dict_outcome)

    dict_outcome = {}

    for d in list_dict_outcome:
        for key, value in d.items():
            if key not in dict_outcome:
                dict_outcome[key] = []
            dict_outcome[key].append(value)

    path = f"{store_code}outcome.csv"

    with open(path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict_outcome.keys())
        writer.writerows(zip(*dict_outcome.values()))




##### OPTIMIZATION MODEL FUNCTIONS #####

def get_opti_matrices(D_out):
    A_latest = np.zeros((D_out, D_out))
    for i in range(D_out):
        for j in range(D_out):
            if i == j:
                if i > 0:
                    A_latest[i, j] = 1
    A_first = np.zeros((D_out, D_out))
    A_last = np.zeros((D_out, D_out))
    A_last[D_out - 1, D_out - 1] = 1
    A_first[0, 0] = 1
    a_first = np.zeros(D_out)
    a_last = np.zeros(D_out)
    a_first[0] = 1
    a_last[D_out - 1] = 1
    A_shift = np.zeros((D_out, D_out))
    for i in range(D_out):
        for j in range(D_out):
            if i == j + 1:
                if i > 0:
                    A_shift[i, j] = 1

    return A_first, A_last, A_latest, A_shift, a_first, a_last

def get_breakpoints(params_dict):
    nb_degr_levels = params_dict['nb_degr_levels']
    max_soc = params_dict['max_soc']
    min_soc = params_dict['min_soc']

    degr_exp = params_dict['degr_exp']
    degr_fact = params_dict['degr_fact']

    soc_breakpoints = np.zeros(nb_degr_levels + 1)
    degr_breakpoints = np.zeros(nb_degr_levels + 1)

    for i in range(nb_degr_levels + 1):
        soc_breakpoints[i] = min_soc + (max_soc - min_soc) / nb_degr_levels * i
        degr_breakpoints[i] = degr_fact * (1 - i / nb_degr_levels) ** (degr_exp)

    return soc_breakpoints, degr_breakpoints

def get_broadcasted_soc_breakpoints(soc_breakpoints, degr_breakpoints, params_dict):
    D_out = params_dict['lookahead']

    bc_soc_bp = np.zeros((D_out, soc_breakpoints.shape[0]))
    bc_degr_bp = np.zeros((D_out, soc_breakpoints.shape[0]))
    bc_fract = np.zeros((D_out, soc_breakpoints.shape[0] - 1))

    for la in range(D_out):
        for bp in range(soc_breakpoints.shape[0]):
            bc_soc_bp[la, bp] = soc_breakpoints[bp]
            bc_degr_bp[la, bp] = degr_breakpoints[bp]
            if bp < soc_breakpoints.shape[0] - 1:
                bc_fract[la, bp] = (degr_breakpoints[bp + 1] - degr_breakpoints[bp]) / (
                            soc_breakpoints[bp + 1] - soc_breakpoints[bp])

    return bc_soc_bp, bc_degr_bp, bc_fract

def interpolate_cds(soc, soc_breakpoints, degr_breakpoints):
    nb_levels = soc_breakpoints.shape[0] - 1
    epsilon = 1e-5

    for lvl in range(nb_levels):
        if soc > soc_breakpoints[lvl + 1] + epsilon:
            pass
        else:
            cds = degr_breakpoints[lvl] + (degr_breakpoints[lvl + 1] - degr_breakpoints[lvl]) / (
            (soc_breakpoints[lvl + 1] - soc_breakpoints[lvl])) * (soc - soc_breakpoints[lvl])
            break

    if 'cds' not in locals():
        print(f"SoC: {soc};SoC breakpoints: {soc_breakpoints}")
        cds = degr_breakpoints[nb_levels]

    return cds

def get_training_labels(train_lab,val_lab,test_lab,decomp='none'):

    if decomp == 'none':
        train_lab_training = train_lab
        val_lab_training = val_lab
        test_lab_training = test_lab
    else:
        train_lab_training = decompose(train_lab,decomp)
        val_lab_training = decompose(val_lab, decomp)
        test_lab_training = decompose(test_lab,decomp)


    return train_lab_training,val_lab_training,test_lab_training

def extend_price(price,params_dict,eff=True):

    lookahead = price.shape[0]

    if eff:
        eff_d = params_dict['eff_d']
        eff_c = params_dict['eff_c']
    else:
        eff_d = 1
        eff_c = 1

    extended_price = np.zeros(3*lookahead+1)
    for la in range(lookahead):
        extended_price[la] = price[la]*eff_d
        extended_price[lookahead+la] = -price[la]/eff_c

    return extended_price

def get_profit(net,input_parameters,model_type,input_features,input_labels,decomp):

    optimal_schedule = get_optimal_schedule(net,input_parameters,input_features)

    price_act_ext = np.zeros((input_labels.shape[0], input_labels.shape[1] * 3 + 1))
    for i in range(input_labels.shape[0]):
        price_act_ext[i, :] = extend_price(input_labels[i, :],input_parameters,eff=True)

    profit_per_timestep = np.multiply(price_act_ext,optimal_schedule)

    return np.sum(profit_per_timestep)

def get_profit_prices(input_parameters,price_fc,price_act):

    price_act_ext = np.zeros((price_act.shape[0], price_act.shape[1] * 3 + 1))
    for i in range(price_act.shape[0]):
        price_act_ext[i, :] = extend_price(price_act[i, :],input_parameters,eff=True)

    optimal_schedule = calc_optimal_schedule(price_fc, input_parameters)
    profit_per_timestep = np.multiply(price_act_ext,optimal_schedule)

    return np.sum(profit_per_timestep)

def calc_optimal_schedule(prices,OP_params_dict,extended_price=False,mode='matrix'):
    n_examples = prices.shape[0]
    lookahead = OP_params_dict['lookahead']
    optimal_schedule = np.zeros((n_examples,3*lookahead+1))

    for ex in range(n_examples):
        #print(ex)
        if extended_price or mode == 'explicit':
            price = prices[ex,:]
        else:
            price = extend_price(prices[ex,:],OP_params_dict,eff = True)
        if mode == 'matrix':
            prob,x = opti_da_cvxpy(OP_params_dict,price)
        elif mode == 'explicit':
            prob,x,d = opti_problem_da_explicit(OP_params_dict,price)


        prob.solve(solver='GUROBI', verbose=False)
        #results = opt.solve(prob, tee=False)
        optimal_schedule[ex,:] = x.value

    return optimal_schedule

def opti_da_cvxpy(params_dict, prices):

    A,b = get_full_matrix_problem_DA(params_dict)


    lookahead = params_dict['lookahead']


    x = cp.Variable(lookahead*3+1)

    objective = cp.Maximize(prices@x)
    constraints=[A@x>=b]

    prob=cp.Problem(objective=objective,constraints=constraints)

    return prob,x

def get_optimal_schedule(net,input_parameters,input_features):

    price_fc = input_features @ np.transpose(net)
    optimal_schedule = calc_optimal_schedule(price_fc[:, 0:24]/input_parameters['eff_d'], input_parameters)

    return optimal_schedule

def get_full_matrix_problem_DA(params_dict):
    #Current definition assumes cyclic boundary conditions
    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']

    A = np.zeros((7 * D_out + 4+2, 3 * D_out + 1))
    b = np.zeros(7 * D_out + 4+2)

    start_d = 0
    start_c = D_out
    start_soc = 2 * D_out

    # positivity constraints
    for t in range(D_out):
        A[t + start_d, t + start_d] = 1
        A[t + start_c, t + start_c] = 1
        A[t + start_soc, t + start_soc] = 1
    A[t + start_soc + 1, t + start_soc + 1] = 1

    # Constrain max power
    start_constrain_max_power = 3 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_max_power, t + start_d] = -1
        A[t + start_constrain_max_power, t + start_c] = -1
        b[t + start_constrain_max_power] = -max_charge

    # Constrain max soc
    start_constrain_soc = 4 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_soc, t + start_soc] = -1
        b[t + start_constrain_soc] = -soc_max
    A[t + start_constrain_soc + 1, t + start_soc + 1] = -1
    b[t + start_constrain_soc + 1] = -soc_max

    # SoC update
    start_soc_update_pos = 5 * D_out + 2
    start_soc_update_neg = 6 * D_out + 3
    for t in range(D_out):
        if t == 0:
            A[t + start_soc_update_pos, t + start_soc] = 1
            b[t + start_soc_update_pos] = soc_0

            A[t + start_soc_update_neg, t + start_soc] = -1
            b[t + start_soc_update_neg] = -soc_0

        else:
            A[t + start_soc_update_pos, t + start_soc] = 1
            A[t + start_soc_update_pos, t - 1 + start_soc] = -1
            A[t + start_soc_update_pos, t - 1 + start_d] = 1#/eff_d
            A[t + start_soc_update_pos, t - 1 + start_c] = -1#*eff_c

            A[t + start_soc_update_neg, t + start_soc] = -1
            A[t + start_soc_update_neg, t - 1 + start_soc] = 1
            A[t + start_soc_update_neg, t - 1 + start_d] = -1#/eff_d
            A[t + start_soc_update_neg, t - 1 + start_c] = 1#*eff_c

        A[t + start_soc_update_pos + 1, t + start_soc + 1] = 1
        A[t + start_soc_update_pos + 1, t - 1 + start_soc + 1] = -1
        A[t + start_soc_update_pos + 1, t - 1 + start_d + 1] = 1#/eff_d
        A[t + start_soc_update_pos + 1, t - 1 + start_c + 1] = -1#*eff_c

        A[t + start_soc_update_neg + 1, t + start_soc + 1] = -1
        A[t + start_soc_update_neg + 1, t - 1 + start_soc + 1] = 1
        A[t + start_soc_update_neg + 1, t - 1 + start_d + 1] = -1#/eff_d
        A[t + start_soc_update_neg + 1, t - 1 + start_c + 1] = 1#*eff_c

    A[t+start_soc_update_neg + 2, t + start_soc+1] = 1
    b[t + start_soc_update_neg + 2] = soc_0

    A[t+start_soc_update_neg + 3, t + start_soc+1] = -1
    b[t + start_soc_update_neg + 3] = -soc_0


    return A, b

def opti_problem_da_explicit(params_dict,price_fc):



    lookahead = params_dict['lookahead']
    eff = params_dict['eff_d']
    min_soc = params_dict['min_soc']
    max_soc = params_dict['max_soc']
    cyclic_bc = params_dict['cyclic_bc']
    soc_0 = params_dict['soc_0']
    max_charge=params_dict['max_charge']




    d = cp.Variable(lookahead)
    c = cp.Variable(lookahead)
    soc = cp.Variable(lookahead+1)
    variables_combined = cp.hstack([c,d,soc])


    constraints=[]

    constraints.append(soc[0] == soc_0)

    for la in range(lookahead):
        constraints.append(c[la] >= 0)
        constraints.append(d[la] >= 0)
        constraints.append(soc[la] >= min_soc)
        constraints.append(soc[la] <= max_soc)
        constraints.append(c[la]+d[la] <= max_charge)
        constraints.append(soc[la+1] == soc[la] + c[la] - d[la])

    if cyclic_bc:
        constraints.append(soc[lookahead] == soc_0)

    obj = cp.Maximize(price_fc@(d*eff-c/eff))

    prob = cp.Problem(objective=obj,constraints=constraints)


    return prob,variables_combined,d

def opti_problem(params_dict):
    # Retrieve optimization parameters
    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']
    gamma = params_dict['gamma']
    cyclic_bc = params_dict['cyclic_bc']

    degradation = params_dict['degradation']
    inv_cost = params_dict['inv_cost']

    # Construct matrices to define optimization problem
    A_first, A_last, A_latest, A_shift, a_first, a_last = get_opti_matrices(D_out)

    e_d = cp.Variable(D_out)
    e_c = cp.Variable(D_out)
    soc = cp.Variable(D_out)
    degr_cost = cp.Variable(D_out)

    net_discharge = cp.Variable(D_out)
    price = cp.Parameter(D_in)

    constraints = [e_d >= 0,
                   e_c >= 0,
                   soc >= soc_min,
                   soc <= soc_max,
                   e_d + e_c <= max_charge,
                   net_discharge == e_d * eff_d - e_c / eff_c,
                   A_latest @ soc == A_shift @ soc + A_latest @ (e_c - e_d),
                   A_first @ soc == soc_0 * a_first + A_first @ (e_c - e_d)]

    if degradation:

        degr_levels = params_dict['nb_degr_levels']

        soc_breakpoints, degr_breakpoints = get_breakpoints(params_dict)
        bc_soc_bps, bc_degr_bps, bc_fract = get_broadcasted_soc_breakpoints(soc_breakpoints, degr_breakpoints,
                                                                            params_dict)
        cds_0 = interpolate_cds(soc_0, soc_breakpoints, degr_breakpoints)

        degr_cost = cp.Variable(D_out)
        cds = cp.Variable(D_out)
        soc_per_level = cp.Variable(shape=(D_out, degr_levels))
        degr_level_bool = cp.Variable(shape=(D_out, degr_levels), boolean=True)

        constraints.append(degr_cost >= 0)
        constraints.append(A_latest @ degr_cost >= inv_cost * soc_max * (A_latest @ cds - A_shift @ cds))
        constraints.append(A_first @ degr_cost >= inv_cost * soc_max * (A_first @ cds - cds_0 * a_first))

        constraints.append(cds == cp.sum(
            expr=cp.multiply(degr_level_bool, bc_degr_bps[:, :-1]) + cp.multiply(bc_fract, soc_per_level - cp.multiply(
                degr_level_bool, bc_soc_bps[:, :-1])), axis=1)
                           )

        constraints.append(cp.sum(degr_level_bool, axis=1) == 1)
        constraints.append(cp.sum(soc_per_level, axis=1) == soc)

        constraints.append(cp.multiply(degr_level_bool, bc_soc_bps[:, :-1]) <= soc_per_level)
        constraints.append(cp.multiply(degr_level_bool, bc_soc_bps[:, 1:]) >= soc_per_level)

        objective = cp.Minimize(-cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c) - degr_cost))


    elif gamma != 0:
        constraints.append(degr_cost == 0)
        id = np.identity(D_out)
        #objective = cp.Minimize(
        #    -cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c)) + gamma * cp.quad_form(e_d * eff_d - e_c / eff_c,id))

        #objective = cp.Minimize(-price @ net_discharge + gamma * cp.quad_form(net_discharge,id)) #old
        #objective = cp.Minimize(-price @ net_discharge + gamma * cp.norm(net_discharge,2))

        if params_dict['smoothing'] == 'log-barrier':

            log_barrier = 0

            # Loop through all constraints and add log-barrier for inequalities
            for constr in constraints:
                if isinstance(constr,cp.constraints.Inequality):
                    log_barrier -= gamma * cp.sum(cp.log(-constr.expr))

            objective = cp.Minimize(-price @ net_discharge + log_barrier)

        elif params_dict['smoothing'] == 'quadratic':

            #objective = cp.Minimize(-price @ net_discharge + gamma * cp.norm(net_discharge,2))
            ##### TODO cp.norm(.,2) does not seem to give expected results, check #####
            #objective = cp.Minimize(-price @ net_discharge + gamma * cp.quad_form(net_discharge,id)) #old
            objective = cp.Minimize(-price @ net_discharge + gamma * cp.sum_squares(net_discharge)) #old


        else:
            sys.exit('Invalid smoothing term')

    else:

        constraints.append(degr_cost == 0)
        objective = cp.Minimize(-cp.sum(cp.multiply(price, e_d * eff_d - e_c / eff_c)))

    if cyclic_bc:
        constraints.append(A_last@soc == soc_0 * a_last)


    prob = cp.Problem(objective=objective, constraints=constraints)

    return prob, [price], [net_discharge,e_d,e_c,soc,degr_cost]


##### DATA PRE-PROCESSING #####

import sys
import pandas as pd
import datetime as dt


#Data reading, feature/label split, forward transformation

def preprocess_data(data_dict):
    #Retrieve relevant parameters from dictionary
    loc_data = data_dict['loc_data']
    scale_mode = data_dict['scale_mode']
    scale_base = data_dict['scale_base']
    cols_features = data_dict['feat_cols']
    col_label_price = data_dict['col_label_price']
    col_label_fc_price = data_dict['col_label_fc_price']
    lookahead = data_dict['lookahead']
    days_train = data_dict['days_train']
    last_ex_test = data_dict['last_ex_test']
    train_share = data_dict['train_share']
    cols_no_centering = data_dict['cols_no_centering']

    #Read data in dataframe and scale
    data_all = get_data_pandas(loc_data=loc_data)
    #data_all_scaled = scale_df(df=data_all,mode=scale_mode,base=scale_base)
    data_all_scaled,stdev_yhat = scale_df_new(data_all,scale_mode,cols_no_centering)
    data_dict["stdev_yhat"] = stdev_yhat
    #Get features and prices in numpy
    features = data_all[cols_features].to_numpy()
    scaled_features = data_all_scaled[cols_features].to_numpy()
    prices = data_all[col_label_price].to_numpy()
    forecasted_prices = data_all[col_label_fc_price].to_numpy()
    forecasted_prices_scaled = data_all_scaled[col_label_fc_price].to_numpy()

    #Select scaled or unscaled features
    if scale_mode == 'none':
        features_for_training = features
    else:
        features_for_training = scaled_features

    #Re-organize features and labels according to lookahead horizon
    forward_features, forward_prices = get_forward_features_labels(features_for_training, prices, lookahead)
    _, forward_fc_price = get_forward_features_labels(features_for_training, forecasted_prices, lookahead)
    _, forward_fc_price_scaled = get_forward_features_labels(features_for_training, forecasted_prices_scaled, lookahead)


    def get_indices(last_ex_test,days_train,train_share,tot_n_examples,mode):

        idx_test = [-(i+1) for i in range(last_ex_test)]


        start = tot_n_examples-last_ex_test-days_train

        if mode == 'separate':
            idx_train = [i for i in range(round(days_train*train_share))]
            idx_val = [round(days_train*train_share) + i for i in range(days_train - round(days_train*train_share))]

        elif mode == 'alternating':

            idx_train = [0]
            idx_val = [1]

            for i in range(days_train-2):
                share = len(idx_train)/(len(idx_train)+len(idx_val))

                if share < train_share:
                    idx_train.append(i+2)
                else:
                    idx_val.append(i+2)

        elif mode == 'alt_test':

            idx_train = [i for i in range(int(round(days_train*train_share)))]
            idx_val = [int(round(days_train*train_share))+1+2*i for i in range(int(round(last_ex_test/2)-1))]
            idx_test = [int(round(days_train * train_share))+1 + 2 * (i - 1) + 1 for i in range(int(round(last_ex_test / 2) - 1))]

            idx_test = [idx + start for idx in idx_test]

        idx_train = [idx + start for idx in idx_train]
        idx_val = [idx + start for idx in idx_val]


        return idx_train,idx_val,idx_test



    indices_train,indices_val,indices_test = get_indices(last_ex_test,days_train,train_share,forward_features.shape[0],data_dict['val_split_mode'])

    features_train = forward_features[indices_train,:]
    features_val = forward_features[indices_val,:]
    features_test = forward_features[indices_test,:]

    price_train = forward_prices[indices_train,:]
    price_val = forward_prices[indices_val,:]
    price_test = forward_prices[indices_test,:]

    price_fc_train = forward_fc_price[indices_train,:]
    price_fc_val = forward_fc_price[indices_val,:]
    price_fc_test = forward_fc_price[indices_test,:]

    price_fc_train_scaled = forward_fc_price_scaled[indices_train, :]
    price_fc_val_scaled = forward_fc_price_scaled[indices_val, :]
    price_fc_test_scaled = forward_fc_price_scaled[indices_test, :]

    return features_train, features_val, features_test, price_train, price_val, price_test, [price_fc_train,price_fc_val, price_fc_test], [price_fc_train_scaled,price_fc_val_scaled,price_fc_test_scaled]


def get_data_pandas(loc_data):
    data_forecast = pd.read_csv(loc_data + 'forecast_df.csv')
    data = pd.read_csv(loc_data + 'X_df_ds.csv')

    data['dt'] = data['ds'].str[:19]
    # data_other['h'] = pd.to_timedelta(data_other['ds'].str[-4].astype(int),'h')
    # data_other['Datetime'] = pd.to_datetime(data_other['dt'])+pd.DateOffset(hours=data_other['h'])
    data['Datetime'] = pd.to_datetime(data['dt'])  # + data_other['h']
    data['date'] = data['Datetime'].dt.date

    data_forecast['dt'] = data_forecast['ds'].str[:19]
    data_forecast['Datetime'] = pd.to_datetime(data_forecast['dt'])
    data_forecast['date'] = data_forecast['Datetime'].dt.date

    dates_rem = [dt.date(2019, 3, 18), dt.date(2019, 3, 31), dt.date(2020, 3, 29), dt.date(2021, 3, 28),
                 dt.date(2022, 3, 27), dt.date(2023, 3, 26),
                 dt.date(2019, 10, 27), dt.date(2020, 10, 25), dt.date(2021, 10, 31), dt.date(2022, 10, 30)]

    data = data[~(data['date'].isin(dates_rem))][1:]
    data_forecast = data_forecast[~(data_forecast['date'].isin(dates_rem))]

    data_all = pd.merge(left=data_forecast, right=data, on='Datetime', how='inner')

    return data_all

def scale_df(df,mode,base=False):

    df_scaled = copy.deepcopy(df)

    if not isinstance(base, str):
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_scaled[col] = scale_column(df[col], mode)
    else:
        stdev_base = df[base].std()
        for col in df.columns:
            if col != base and pd.api.types.is_numeric_dtype(df[col]):
                df_scaled[col] = scale_column(df[col], mode) * stdev_base

    # for col in df.columns:
    #     if pd.api.types.is_numeric_dtype(df[col]):
    #         df_scaled[col] = scale_column(column=df[col],mode=mode)

    return df_scaled


def scale_df_new(df,mode,cols_no_centering):

    df_scaled = copy.deepcopy(df)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in cols_no_centering:
                df_scaled[col] = scale_column(df[col], "stand_no_centering")
            else:
                df_scaled[col] = scale_column(df[col], mode)

    stdev_yhat = 0
    if "y_hat" in df.columns:
        stdev_yhat = np.std(df["y_hat"])

    return df_scaled,stdev_yhat

def scale_column(column,mode):

    if mode == 'stand':
        scaled_column = (column-column.mean())/column.std()
    elif mode == 'norm':
        scaled_column = (column-column.min())/(column.max()-column.min())
    elif mode == 'none':
        scaled_column = column
    elif mode == 'stand_no_centering':
        scaled_column = column/column.std()
    else:
        sys.exit('invalid scaling mode')
    return scaled_column

def get_forward_features_labels(features,labels,lookahead):
    n_rows = features.shape[0]
    n_features = features.shape[1]
    n_examples = int(n_rows/lookahead)


    forward_features=np.zeros((n_examples,n_features*lookahead))
    forward_labels = np.zeros((n_examples,lookahead))

    for ex in range(n_examples):
        for la in range(lookahead):
            forward_labels[ex,la] = labels[lookahead*ex+la]
            for feat in range(n_features):
                forward_features[ex,n_features*la+feat] = features[lookahead*ex+la,feat]
    return forward_features,forward_labels


#Retrieve correct labels

def preprocess_labels(type, price_train, price_val, price_test, OP_params_dict, data_dict):

    if data_dict["scale_price"]:
        price_train = price_train/data_dict["stdev_yhat"]
        price_val = price_val/data_dict["stdev_yhat"]
        price_test = price_test/data_dict["stdev_yhat"]

    if type == 'price':
        labels_train = price_train
        labels_val = price_val
        labels_test = price_test
    elif type == 'price_schedule':
        sched_train = calc_optimal_schedule(price_train,OP_params_dict,mode='explicit')
        #TODO: check impact of factor size: this way probably augments the size twice in new cases of train/val set?
        labels_train = [price_train,sched_train]
        labels_val = price_val
        labels_test = price_test

    return labels_train,labels_val,labels_test

def retrieve_optimal_schedule_train_test(train_prices,validation_prices, OP_params_dict,mode='matrix'):

    #TODO: remove function?

    optimized_train_sched = calc_optimal_schedule(train_prices,OP_params_dict,mode=mode)
    optimized_val_sched = calc_optimal_schedule(validation_prices,OP_params_dict,mode=mode)

    return optimized_train_sched,optimized_val_sched


#Convert to tensor and train dataset
def get_train_validation_tensors(train_features, train_labels, validation_features, validation_labels, test_features, test_labels, model_type, type_train_labels):
    if model_type == 'edRNN':
        tensor_train_features_e = torch.from_numpy(train_features[0]).float()
        tensor_train_features_d = torch.from_numpy(train_features[1]).float()
        tensor_train_features = [tensor_train_features_e, tensor_train_features_d]

        if type_train_labels == 'price_schedule':
            tensor_train_labels = [torch.from_numpy(train_labels[0]).float(),torch.from_numpy(train_labels[1]).float()]
        else:
            tensor_train_labels = torch.from_numpy(train_labels).float()

        tensor_validation_features_e = torch.from_numpy(validation_features[0]).float()
        tensor_validation_features_d = torch.from_numpy(validation_features[1]).float()
        tensor_validation_features = [tensor_validation_features_e, tensor_validation_features_d]


        tensor_validation_labels = torch.from_numpy(validation_labels).float()

        tensor_test_features_e = torch.from_numpy(test_features[0]).float()
        tensor_test_features_d = torch.from_numpy(test_features[1]).float()
        tensor_test_features = [tensor_test_features_e, tensor_test_features_d]

        tensor_test_labels = torch.from_numpy(test_labels).float()

    else:
        tensor_train_features = torch.from_numpy(train_features).float()
        if type_train_labels == 'price_schedule':
            tensor_train_labels = [torch.from_numpy(train_labels[0]).float(),torch.from_numpy(train_labels[1]).float()]
        else:
            tensor_train_labels = torch.from_numpy(train_labels).float()
        tensor_validation_features = torch.from_numpy(validation_features).float()
        tensor_validation_labels = torch.from_numpy(validation_labels).float()
        tensor_test_features = torch.from_numpy(test_features).float()
        tensor_test_labels = torch.from_numpy(test_labels).float()

    return [tensor_train_features, tensor_train_labels, tensor_validation_features, tensor_validation_labels, tensor_test_features, tensor_test_labels]

def set_tensors_to_device(list_tensors,dev):
    # Set list of tensors to specified device

    global_list = []

    for tensors in list_tensors:
        if type(tensors) is list:
            new_entry = [tensors[0].to(dev),tensors[1].to(dev)]
        else:
            new_entry = tensors.to(dev)
        global_list.append(new_entry)

    return global_list

def get_train_Dataset(model_type, type_train_labels, train_feat, train_lab_training):

    if model_type == 'edRNN':
        if type_train_labels == 'price_schedule':
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training[0],
                                                           train_lab_training[1])
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training)
    else:
        if type_train_labels == 'price_schedule':
            train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training[0], train_lab_training[1])
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training)

    return train_Dataset

# Converting dataframe raw data to usable arrays





def split_df_features_labels_arrays(df, quantiles, lookahead, nb_forecast_qhs_included, feature_scaling=False,
                                    feature_scaler=None, model_type='NN', training_type='FC_SI', incl_prev_price=False,
                                    label_col='adjusted_imb_price_alpha', prev_imb_price_qh=0, price_split=False):
    df = df.copy(deep=True)
    n_quantiles = len(quantiles)

    if price_split:
        df = replace_imb_price_split(df)

    # Drop columns of the dataframe based on training type, lookahead
    cols_to_delete = cols_to_delete_per_type(training_type, incl_prev_price, len(quantiles))
    if label_col in cols_to_delete:
        cols_to_delete.remove(label_col)
    df.drop(cols_to_delete, axis=1, inplace=True)

    cols_to_delete = cols_to_delete_la(n_quantiles, lookahead, nb_forecast_qhs_included)
    df.drop(cols_to_delete, axis=1, inplace=True)

    # Sort dataframe and ensure the Datetime column values are of type datetime
    df_sorted = sort_dataframe(df)

    # Find the indices for which all the quarter hours of the lookahead and lookback are also included in the dataframe, hence which can be included in the training data
    list_of_training_indices = get_training_indices(df_sorted, lookahead, prev_imb_price_qh)

    # Split dataframe in dataframe(s) for labels and features
    # Distinction is made between the forecasted SI and other features as the former are included for the entire
    # lookahead horizon in one row of the original dataframe, whereas the other features should be collected from
    # the rows corresponding to their respective quarter hour
    df_features = df_sorted.drop(columns=['Datetime', 'adjusted_imb_price_alpha'], axis=1)

    if feature_scaling:
        cols = list(df_features.columns)
        if feature_scaler == None:
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_features.to_numpy())
        else:
            scaler = feature_scaler
            df_scaled = scaler.transform(df_features.to_numpy())

        df_features = pd.DataFrame(df_scaled, columns=cols)

    df_features_except_FC_SI = df_features.drop(columns=[str(i + 1) for i in range(n_quantiles * lookahead)], axis=1)
    df_labels = df_sorted[label_col]

    # Match data from correct rows to the different lookahead values and return arrays for features and labels
    if model_type == 'NN' or model_type == 'LR':
        features, labels = get_features_labels_arrays_numpy_nn(list_of_training_indices, lookahead, n_quantiles,
                                                               df_labels, df_features, df_features_except_FC_SI,
                                                               prev_imb_price_qh)
    elif model_type == 'vRNN':
        features, labels = get_features_labels_arrays_numpy_vRNN(list_of_training_indices, lookahead, n_quantiles,
                                                                 df_labels, df_features, df_features_except_FC_SI,
                                                                 prev_imb_price_qh)
    elif model_type == 'edRNN':
        features, labels = get_features_labels_arrays_numpy_edRNN(list_of_training_indices, lookahead, n_quantiles,
                                                                  df_labels, df_features, df_features_except_FC_SI,
                                                                  prev_imb_price_qh)

    dict = {'features': features,
            'labels': labels}

    if feature_scaling:
        dict['scaler'] = scaler

    return dict


def get_features_labels_arrays_numpy_nn(list_of_training_indices, lookahead, n_quantiles, df_labels, df_features,
                                        df_features_except_FC_SI, prev_imb_price_qhs):
    labels_array_short = df_labels.to_numpy()
    features_array_short = df_features.to_numpy()
    features_except_FC_SI_array_short = df_features_except_FC_SI.to_numpy()

    n_instances = len(list_of_training_indices)
    n_features_except_FC_SI = df_features_except_FC_SI.shape[1]
    n_features = n_features_except_FC_SI + n_quantiles

    array_features = np.zeros((n_instances, n_features * lookahead + prev_imb_price_qhs))
    array_labels = np.zeros((n_instances, lookahead))

    for i in range(len(list_of_training_indices)):
        index = list_of_training_indices[i]
        for la in range(lookahead):
            array_labels[i, la] = labels_array_short[index + la]
            array_features[i,
            la * n_features:la * n_features + n_features_except_FC_SI] = features_except_FC_SI_array_short[index + la,
                                                                         :]
            for quant in range(n_quantiles):
                array_features[i, la * n_features + n_features_except_FC_SI + quant] = features_array_short[
                    index, la * n_quantiles + quant + features_except_FC_SI_array_short.shape[1]]
            for prev in range(prev_imb_price_qhs):
                array_pos = prev_imb_price_qhs - prev
                array_features[i, -(array_pos)] = labels_array_short[index - (prev + 1)]

    return array_features, array_labels


def get_features_labels_arrays_numpy_vRNN(list_of_training_indices, lookahead, n_quantiles, df_labels, df_features,
                                          df_features_except_FC_SI, prev_imb_price_qhs):
    labels_array_short = df_labels.to_numpy()
    features_array_short = df_features.to_numpy()
    features_except_FC_SI_array_short = df_features_except_FC_SI.to_numpy()

    n_instances = len(list_of_training_indices)
    n_features_except_FC_SI = df_features_except_FC_SI.shape[1]
    n_features = n_features_except_FC_SI + n_quantiles

    array_features = np.zeros((n_instances, lookahead, n_features + prev_imb_price_qhs))
    array_labels = np.zeros((n_instances, lookahead))

    for i in range(len(list_of_training_indices)):
        index = list_of_training_indices[i]
        for la in range(lookahead):
            array_labels[i, la] = labels_array_short[index + la]
            array_features[i, la, 0:n_features_except_FC_SI] = features_except_FC_SI_array_short[index + la, :]
            for quant in range(n_quantiles):
                array_features[i, la, n_features_except_FC_SI + quant] = features_array_short[
                    index, la * n_quantiles + quant + features_except_FC_SI_array_short.shape[1]]
            for prev in range(prev_imb_price_qhs):
                array_pos = prev_imb_price_qhs - prev
                array_features[i, la, -(array_pos)] = labels_array_short[index - (prev + 1)]

    return array_features, array_labels


def get_features_labels_arrays_numpy_edRNN(list_of_training_indices, lookahead, n_quantiles, df_labels, df_features,
                                           df_features_except_FC_SI, prev_imb_price_qhs):
    labels_array_short = df_labels.to_numpy()
    features_array_short = df_features.to_numpy()
    features_except_FC_SI_array_short = df_features_except_FC_SI.to_numpy()

    n_instances = len(list_of_training_indices)
    n_features_except_FC_SI = df_features_except_FC_SI.shape[1]
    n_features = n_features_except_FC_SI + n_quantiles

    array_features_future = np.zeros((n_instances, lookahead, n_features))
    array_features_past = np.zeros(
        (n_instances, prev_imb_price_qhs, n_features + 1))  # third dimension one more value being the imbalance price
    array_labels = np.zeros((n_instances, lookahead))

    for i in range(len(list_of_training_indices)):
        index = list_of_training_indices[i]
        for la in range(lookahead):
            array_labels[i, la] = labels_array_short[index + la]
            array_features_future[i, la, 0:n_features_except_FC_SI] = features_except_FC_SI_array_short[index + la, :]
            for quant in range(n_quantiles):
                array_features_future[i, la, n_features_except_FC_SI + quant] = features_array_short[
                    index, la * n_quantiles + quant + features_except_FC_SI_array_short.shape[1]]

        for lb in range(prev_imb_price_qhs):
            array_pos = prev_imb_price_qhs - lb
            array_features_past[i, -(array_pos), 0:n_features_except_FC_SI] = features_except_FC_SI_array_short[
                                                                              index - (lb + 1), :]
            for quant in range(n_quantiles):
                array_features_past[i, -(array_pos), n_features_except_FC_SI + quant] = features_array_short[
                    index - (lb + 1), quant + features_except_FC_SI_array_short.shape[1]]
            array_features_past[i, -(array_pos), -1] = labels_array_short[index - (lb + 1)]

    features = [array_features_past, array_features_future]

    return features, array_labels


def replace_imb_price_split(df):
    df['intermediate'] = df['adjusted_imb_price_alpha'] - 1 / 2 * (df['-100MW'] + df['100MW'])
    df.drop(['adjusted_imb_price_alpha'], axis=1, inplace=True)
    df.rename(columns={'intermediate': 'adjusted_imb_price_alpha'}, inplace=True)

    return df


def cols_to_delete_per_type(type, incl_prev_imb_price, n_quantiles):
    if type == 'SI':
        cols_to_delete = ['NRV', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D']
        for col in range(9):
            cols_to_delete += [str(col + 1)]
    elif type == 'FC_SI':
        cols_to_delete = ['SI', 'NRV', 'Vol_GCC_I', 'GUV', 'GDV', 'Vol_GCC_D']
    else:
        print(x)
        print('Error: invalid train type. Please use SI or FC_SI')
        sys.exit()

    if incl_prev_imb_price:
        return cols_to_delete
    else:
        return cols_to_delete + ['adjusted_imb_price_alpha_previous']


def cols_to_delete_la(n_quantiles, lookahead, nb_forecast_qhs_included):
    cols_to_delete = []

    if lookahead < nb_forecast_qhs_included:
        for qh in range(int(nb_forecast_qhs_included - lookahead)):
            for quant in range(n_quantiles):
                cols_to_delete.append(str((lookahead + qh) * n_quantiles + quant + 1))

    return cols_to_delete


def sort_dataframe(dataframe):
    df_sorted = dataframe.sort_values(by='Datetime')
    datetime_format = "%Y-%m-%d %H:%M:%S"
    df_sorted['Datetime'] = pd.to_datetime(df_sorted['Datetime'], format=datetime_format)
    df_sorted.reset_index()

    return df_sorted


def get_training_indices(df, lookahead, prev_imb_price_qhs):
    list_of_indices = []

    for index, row in df.iterrows():
        if (index <= df.shape[0] - lookahead) & (index >= prev_imb_price_qhs):
            if row['Datetime'] + dt.timedelta(minutes=(lookahead - 1) * 15) == df['Datetime'][index + lookahead - 1]:
                list_of_indices.append(index)

    return list_of_indices


def get_features_dict_from_df(df, quantiles, lookahead, nb_forecast_qhs_included, model_type='NN',
                              training_type='FC_SI', incl_prev_price=False, label_col='adjusted_imb_price_alpha',
                              prev_imb_price_qh=0):
    types = ['NN', 'vRNN', 'edRNN']

    features_dict = {}

    for type in types:
        features_dict[type], test_labels = split_df_features_labels_arrays(df, quantiles, lookahead,
                                                                           nb_forecast_qhs_included, False, None, type,
                                                                           training_type, incl_prev_price, label_col,
                                                                           prev_imb_price_qh)

    return features_dict, test_labels


# Splitting data set in train & validation set

def split_features_labels_train_validation(features, labels, amount_of_validation_periods: int, share_validation,
                                           lookahead: int):
    if amount_of_validation_periods <= 1:
        print('Minimum amount of valdiation periods is 2')
        sys.exit()

    total_instances = labels.shape[0]

    start_array_val, end_array_val, start_array_train, end_array_train = get_pos_arrays(amount_of_validation_periods,
                                                                                        total_instances, lookahead,
                                                                                        share_validation)

    return get_arrays_from_positions(features, labels, start_array_val, end_array_val, start_array_train,
                                     end_array_train, amount_of_validation_periods)


def get_lost_instances(amount_of_validation_periods, lookahead):
    if amount_of_validation_periods < 2:
        return amount_of_validation_periods * lookahead
    elif amount_of_validation_periods >= 2:
        return (amount_of_validation_periods - 1) * 2 * lookahead


def get_pos_arrays(amount_of_validation_periods, total_instances, lookahead, share_validation):
    amount_of_training_periods = amount_of_validation_periods - 1

    leftover_instances = total_instances - get_lost_instances(amount_of_validation_periods, lookahead)

    instances_validation = share_validation * leftover_instances

    instance_per_validation_period = round(instances_validation / amount_of_validation_periods)
    instance_per_training_period = round((leftover_instances - instances_validation) / amount_of_training_periods)

    start_array_val = []
    end_array_val = []
    start_array_train = []
    end_array_train = []

    validation_periods_defined = 0

    while validation_periods_defined < amount_of_validation_periods - 1:
        start = validation_periods_defined * (
                    instance_per_validation_period + 2 * lookahead + instance_per_training_period)

        start_array_val.append(start)
        end_array_val.append(start + instance_per_validation_period)
        start_array_train.append(start + instance_per_validation_period + lookahead)
        end_array_train.append(start + instance_per_validation_period + lookahead + instance_per_training_period)

        validation_periods_defined += 1

    start_array_val.append(
        validation_periods_defined * (instance_per_validation_period + 2 * lookahead + instance_per_training_period))
    end_array_val.append(total_instances)

    return start_array_val, end_array_val, start_array_train, end_array_train


def get_arrays_from_positions(features, labels, start_array_val, end_array_val, start_array_train, end_array_train,
                              amount_of_validation_periods):
    labels_val = labels[start_array_val[0]:end_array_val[0]]
    labels_train = labels[start_array_train[0]:end_array_train[0]]

    if type(features) == list:
        features_past = features[0]
        features_future = features[1]

        features_val_past = features_past[start_array_val[0]:end_array_val[0], :]
        features_train_past = features_past[start_array_train[0]:end_array_train[0], :]
        features_val_fut = features_future[start_array_val[0]:end_array_val[0], :]
        features_train_fut = features_future[start_array_train[0]:end_array_train[0], :]

        periods_determined = 1

        while periods_determined < amount_of_validation_periods:

            if periods_determined < amount_of_validation_periods - 1:
                features_train_past = np.concatenate((features_train_past, features_past[
                                                                           start_array_train[periods_determined]:
                                                                           end_array_train[periods_determined], :]))
                features_train_fut = np.concatenate((features_train_fut, features_future[
                                                                         start_array_train[periods_determined]:
                                                                         end_array_train[periods_determined], :]))
                labels_train = np.concatenate(
                    (labels_train, labels[start_array_train[periods_determined]:end_array_train[periods_determined]]))

            features_val_past = np.concatenate((features_val_past, features_past[
                                                                   start_array_val[periods_determined]:end_array_val[
                                                                       periods_determined], :]))
            features_val_fut = np.concatenate((features_val_fut, features_future[
                                                                 start_array_val[periods_determined]:end_array_val[
                                                                     periods_determined], :]))

            labels_val = np.concatenate(
                (labels_val, labels[start_array_val[periods_determined]:end_array_val[periods_determined]]))

            periods_determined += 1

        features_train = [features_train_past, features_train_fut]
        features_val = [features_val_past, features_val_fut]


    else:
        features_val = features[start_array_val[0]:end_array_val[0], :]
        features_train = features[start_array_train[0]:end_array_train[0], :]

        periods_determined = 1

        while periods_determined < amount_of_validation_periods:

            if periods_determined < amount_of_validation_periods - 1:
                features_train = np.concatenate((features_train, features[
                                                                 start_array_train[periods_determined]:end_array_train[
                                                                     periods_determined], :]))
                labels_train = np.concatenate(
                    (labels_train, labels[start_array_train[periods_determined]:end_array_train[periods_determined]]))

            features_val = np.concatenate(
                (features_val, features[start_array_val[periods_determined]:end_array_val[periods_determined], :]))
            labels_val = np.concatenate(
                (labels_val, labels[start_array_val[periods_determined]:end_array_val[periods_determined]]))

            periods_determined += 1

    return features_train, features_val, labels_train, labels_val


##### FUNCTIONS FOR IMBALANCE PRICE CALCULATION #####

def calculate_imbalance_price_from_feature_dict(list_models, list_lookahead, list_model_names, list_model_package,
                                                list_lookbacks, features_dict, col_SI, quantiles_FC_SI, decomp='none',
                                                split_price=False):
    # V2 of calculate_imbalance_price(...)
    # TO DO: implement potential feature scaling / should this be in other method?

    model_type = list_model_names[0]
    try:
        n_qhs = features_dict[model_type].shape[0]
    except:
        try:
            n_qhs = features_dict['edRNN'][0].shape[0]
        except:
            n_qhs = features_dict['NN'].shape[0]

    max_la = max(list_lookahead)
    max_lb = max(list_lookbacks)
    n_models = len(list_models)

    if decomp in ['none', 'dwt_haar']:
        predicted_imb_price = np.zeros((n_models, n_qhs, max_la))
    elif decomp == 'fft':
        predicted_imb_price = np.zeros((n_models, n_qhs, max_la * 2))

    for index in range(n_models):
        model = list_models[index]
        pack = list_model_package[index]
        model_name = list_model_names[index]
        la = list_lookahead[index]
        lb = list_lookbacks[index]

        features_adj = get_features_full_la(model_name, lb, max_lb, features_dict)

        if model == 'NA':
            predicted_imb_price[index, :, :] = calculate_predicted_price_OP(features_adj, max_la, col_SI,
                                                                            quantiles_FC_SI)
        else:
            if pack == 'tf':
                predicted_imb_price[index, :, :] = calculate_predicted_price_model_v2(model_name, model, features_adj,
                                                                                      la, max_la, n_qhs, lb)
            elif pack == 'torch_price':
                predicted_imb_price[index, :, :] = calculated_predicted_price_model_torch(model.to('cpu'), features_adj,
                                                                                          'price')
            elif pack == 'torch_schedule':
                predicted_imb_price[index, :, :] = calculated_predicted_price_model_torch(model.to('cpu'), features_adj,
                                                                                          'schedule')
            elif pack == 'spo_LR':
                predicted_imb_price[index, :, :] = features_adj @ np.transpose(model)

        if split_price:
            for qh in range(n_qhs):
                for la in range(max_la):
                    predicted_imb_price[index, qh, la] += (features_dict['edRNN'][1][qh, la, 9] +
                                                           features_dict['edRNN'][1][qh, la, 10]) / 2

        predicted_imb_price = convert_decomposition(predicted_imb_price, decomp)

    return predicted_imb_price


def convert_decomposition(forecast, decomp):
    if decomp == 'none':
        price = forecast

    elif decomp == 'dwt_haar':
        la = forecast.shape[2]
        price = np.zeros_like(forecast)

        for index in range(forecast.shape[0]):
            for qh in range(forecast.shape[1]):
                price[index, qh, :] = pywt.idwt(forecast[index, qh, 0:int(la / 2)], forecast[index, qh, int(la / 2):],
                                                'haar')

    elif decomp == 'fft':
        la = int(forecast.shape[2] / 2)
        price = np.zeros((forecast.shape[0], forecast.shape[1], la))

        for index in range(forecast.shape[0]):
            for qh in range(forecast.shape[1]):
                real = forecast[index, qh, 0:la]
                imag = forecast[index, qh, la:]

                complex = real + 1j * imag

                price[index, qh, :] = np.real(np.fft.ifft(complex))

    return price


def calculate_predicted_price_OP(features, max_la, col_SI, quantiles_FC_SI):
    n_qhs = features.shape[0]
    feature_cols_per_inst = int(features.shape[1] / max_la)

    predicted_price = np.zeros((n_qhs, max_la))

    for pred in range(max_la):
        start = pred * feature_cols_per_inst
        end = (pred + 1) * feature_cols_per_inst
        reduced_features = features[:, start:end]
        predicted_price[:, pred] = opti_prediction_single(reduced_features, 'FC_SI', col_SI, quantiles_FC_SI)

    return predicted_price


def calculated_predicted_price_model_torch(model, features_adj, model_type='schedule'):
    if isinstance(features_adj, list):
        input_e = torch.from_numpy(features_adj[0]).float()
        input_d = torch.from_numpy(features_adj[1]).float()
        if model_type == 'price':
            predicted_prices = model(input_e, input_d, 'cpu').detach().numpy()
        else:
            predicted_prices = model.price_generator(input_e, input_d, 'cpu').detach().numpy()

    else:
        features_tensor = torch.from_numpy(features_adj).float()
        # predicted_prices = model.lstm(features_tensor).detach().numpy()
        if model_type == 'price':
            predicted_prices = model(features_tensor).detach().numpy()
        else:
            predicted_prices = model.price_generator(features_tensor).detach().numpy()

    return predicted_prices


def calculate_predicted_price_model_v2(model_name, model, features, la, max_la, n_qhs, lookback):
    predicted_price = np.zeros((n_qhs, max_la))

    if la == max_la:

        predicted_price = model.predict(features)

    else:
        nb_predictions = int(max_la / la)

        for pred in range(nb_predictions):
            if model_name == 'NN':
                repeated_feature_cols_per_inst = int((features.shape[1] - lookback) / max_la)
                start = pred * repeated_feature_cols_per_inst * la
                end = (pred + 1) * repeated_feature_cols_per_inst * la
                reduced_features = features[:, start:end]
                if lookback > 0:
                    reduced_features = np.append(reduced_features, features[:, -lookback:],
                                                 axis=1)  # Add the lookback imbalance prices
                predicted_price[:, pred * la:(pred + 1) * la] = model.predict(reduced_features)
            elif model_name == 'vRNN':
                start = pred * la
                end = (pred + 1) * la
                reduced_features = features[:, start:end, :]
                predicted_price[:, start:end] = model.predict(reduced_features)
            elif model_name == 'edRNN':
                start = pred * la
                end = (pred + 1) * la
                reduced_features_future = features[1][:, start:end, :]
                predicted_price[:, start:end] = model.predict([features[0], reduced_features_future])

    return predicted_price


def opti_prediction_single(features, type, col_SI, quantiles_FC_SI):
    n_qhs = features.shape[0]
    calculated_prices = np.zeros(n_qhs)

    if type == 'SI':

        for sample in range(n_qhs):
            val_SI = -features[sample, col_SI]
            calculated_prices[sample] = find_price_from_SI_MO(val_SI, features[sample, 0:col_SI])

    elif type == 'FC_SI':

        quantiles_prob = calc_prob_for_quantiles(quantiles_FC_SI)

        for sample in range(n_qhs):
            quantile_prices_per_sample = []

            for i in range(len(quantiles_FC_SI)):
                val_SI_quantile = -features[sample, col_SI + i]
                quantile_prices_per_sample.append(find_price_from_SI_MO(val_SI_quantile, features[sample, 0:20]))

            calculated_prices[sample] = sum(np.multiply(quantile_prices_per_sample, quantiles_prob))

    return calculated_prices


def find_price_from_SI_MO(SI, merit_order):
    SI = limit_SI(SI)

    activation_level = (SI - (SI % 100)) / 100
    col = int(activation_level + 10)
    return merit_order[col]


def limit_SI(SI):
    if SI >= 1000:
        SI = 999.99
    elif SI <= -1000:
        SI = -999.99

    return SI


def calc_prob_for_quantiles(quantiles):
    prob_for_quantiles = [0] * len(quantiles)

    for i in range(len(prob_for_quantiles)):

        if i == 0:
            lo_lim = 0
        else:
            lo_lim = (quantiles[i - 1] + quantiles[i]) / 2

        if i == len(quantiles) - 1:
            up_lim = 1
        else:
            up_lim = (quantiles[i] + quantiles[i + 1]) / 2

        prob_for_quantiles[i] = up_lim - lo_lim

    return prob_for_quantiles


def get_features_full_la(model_name, lb, max_lb, features_dict):
    if model_name in ['NN', 'OP', 'LR']:

        key = 'NN'

        if lb == max_lb:
            features = features_dict[key][:, :]
        else:
            features = features_dict[key][:, :-(max_lb - lb)]

    elif model_name == 'vRNN':
        if lb == max_lb:
            features = features_dict['vRNN'][:, :, :]
        else:
            features = features_dict['vRNN'][:, :, :-(max_lb - lb)]

    elif model_name == 'edRNN':
        if lb == max_lb:
            features_past = features_dict['edRNN'][0][:, :, :]
        else:
            features_past = features_dict['edRNN'][0][:, :-(max_lb - lb), :]
        features_fut = features_dict['edRNN'][1]
        features = [features_past, features_fut]

    return features


##### SPO Functions #####

def convert_prices(prices, dict):
    eff = dict['eff_d']

    lookahead = prices.shape[1]

    new_prices = np.zeros((prices.shape[0], 3 * lookahead + 1))

    new_prices[:, 0:lookahead] = prices * eff
    new_prices[:, lookahead:2 * lookahead] = -prices / eff

    return new_prices


def convert_schedule(schedule, dict):
    eff = dict['eff_d']
    soc_0 = dict['soc_0']
    lookahead = schedule.shape[1]

    new_schedule = np.zeros((schedule.shape[0], 3 * lookahead + 1))

    for qh in range(schedule.shape[0]):
        new_schedule[qh, 2 * lookahead] = soc_0
        for i in range(lookahead):

            if schedule[qh, i] > 0:
                new_schedule[qh, i] = schedule[qh, i] / eff
            else:
                new_schedule[qh, i + lookahead] = -schedule[qh, i] * eff

            new_schedule[qh, 2 * lookahead + i + 1] = new_schedule[qh, 2 * lookahead + i] - new_schedule[qh, i] + \
                                                      new_schedule[qh, i + lookahead]

    return new_schedule


def get_full_matrix_problem(params_dict):

    ### THIS MAY NOT BE CORRECT --> CHECK JULIA ###

    D_out = params_dict['lookahead']
    D_in = D_out
    eff_c = params_dict['eff_c']
    eff_d = params_dict['eff_d']
    soc_0 = params_dict['soc_0']
    soc_max = params_dict['max_soc']
    soc_min = params_dict['min_soc']
    max_charge = params_dict['max_charge']

    A = np.zeros((7 * D_out + 4, 3 * D_out + 1))
    b = np.zeros(7 * D_out + 4)

    start_d = 0
    start_c = D_out
    start_soc = 2 * D_out

    # positivity constraints
    for t in range(D_out):
        A[t + start_d, t + start_d] = 1
        A[t + start_c, t + start_c] = 1
        A[t + start_soc, t + start_soc] = 1
    A[t + start_soc + 1, t + start_soc + 1] = 1

    # Constrain max power
    start_constrain_max_power = 3 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_max_power, t + start_d] = -1
        A[t + start_constrain_max_power, t + start_c] = -1
        b[t + start_constrain_max_power] = -max_charge

    # Constrain max soc
    start_constrain_soc = 4 * D_out + 1
    for t in range(D_out):
        A[t + start_constrain_soc, t + start_soc] = -1
        b[t + start_constrain_soc] = -soc_max
    A[t + start_constrain_soc + 1, t + start_soc + 1] = -1
    b[t + start_constrain_soc + 1] = -soc_max

    # SoC update
    start_soc_update_pos = 5 * D_out + 2
    start_soc_update_neg = 6 * D_out + 3
    for t in range(D_out):
        if t == 0:
            A[t + start_soc_update_pos, t + start_soc] = 1
            b[t + start_soc_update_pos] = soc_0

            A[t + start_soc_update_neg, t + start_soc] = -1
            b[t + start_soc_update_neg] = -soc_0

        else:
            A[t + start_soc_update_pos, t + start_soc] = 1
            A[t + start_soc_update_pos, t - 1 + start_soc] = -1
            A[t + start_soc_update_pos, t - 1 + start_d] = 1
            A[t + start_soc_update_pos, t - 1 + start_c] = -1

            A[t + start_soc_update_neg, t + start_soc] = -1
            A[t + start_soc_update_neg, t - 1 + start_soc] = 1
            A[t + start_soc_update_neg, t - 1 + start_d] = -1
            A[t + start_soc_update_neg, t - 1 + start_c] = 1

        A[t + start_soc_update_pos + 1, t + start_soc + 1] = 1
        A[t + start_soc_update_pos + 1, t - 1 + start_soc + 1] = -1
        A[t + start_soc_update_pos + 1, t - 1 + start_d + 1] = 1
        A[t + start_soc_update_pos + 1, t - 1 + start_c + 1] = -1

        A[t + start_soc_update_neg + 1, t + start_soc + 1] = -1
        A[t + start_soc_update_neg + 1, t - 1 + start_soc + 1] = 1
        A[t + start_soc_update_neg + 1, t - 1 + start_d + 1] = -1
        A[t + start_soc_update_neg + 1, t - 1 + start_c + 1] = 1

    return A, b


# def get_full_matrix_problem_DA(params_dict):
#     D_out = params_dict['lookahead']
#     D_in = D_out
#     eff_c = params_dict['eff_c']
#     eff_d = params_dict['eff_d']
#     soc_0 = params_dict['soc_0']
#     soc_max = params_dict['max_soc']
#     soc_min = params_dict['min_soc']
#     max_charge = params_dict['max_charge']
#
#     A = np.zeros((7 * D_out + 4, 3 * D_out + 1))
#     b = np.zeros(7 * D_out + 4)
#
#     start_d = 0
#     start_c = D_out
#     start_soc = 2 * D_out
#
#     # positivity constraints
#     for t in range(D_out):
#         A[t + start_d, t + start_d] = 1
#         A[t + start_c, t + start_c] = 1
#         A[t + start_soc, t + start_soc] = 1
#     A[t + start_soc + 1, t + start_soc + 1] = 1
#
#     # Constrain max power
#     start_constrain_max_power = 3 * D_out + 1
#     for t in range(D_out):
#         A[t + start_constrain_max_power, t + start_d] = -1
#         A[t + start_constrain_max_power, t + start_c] = -1
#         b[t + start_constrain_max_power] = -max_charge
#
#     # Constrain max soc
#     start_constrain_soc = 4 * D_out + 1
#     for t in range(D_out):
#         A[t + start_constrain_soc, t + start_soc] = -1
#         b[t + start_constrain_soc] = -soc_max
#     A[t + start_constrain_soc + 1, t + start_soc + 1] = -1
#     b[t + start_constrain_soc + 1] = -soc_max
#
#     # SoC update
#     start_soc_update_pos = 5 * D_out + 2
#     start_soc_update_neg = 6 * D_out + 3
#     for t in range(D_out):
#         if t == 0:
#             A[t + start_soc_update_pos, t + start_soc] = 1
#             b[t + start_soc_update_pos] = soc_0
#
#             A[t + start_soc_update_neg, t + start_soc] = -1
#             b[t + start_soc_update_neg] = -soc_0
#
#         else:
#             A[t + start_soc_update_pos, t + start_soc] = 1
#             A[t + start_soc_update_pos, t - 1 + start_soc] = -1
#             A[t + start_soc_update_pos, t - 1 + start_d] = 1 / eff_d
#             A[t + start_soc_update_pos, t - 1 + start_c] = -1 * eff_c
#
#             A[t + start_soc_update_neg, t + start_soc] = -1
#             A[t + start_soc_update_neg, t - 1 + start_soc] = 1
#             A[t + start_soc_update_neg, t - 1 + start_d] = -1 / eff_d
#             A[t + start_soc_update_neg, t - 1 + start_c] = 1 * eff_c
#
#         A[t + start_soc_update_pos + 1, t + start_soc + 1] = 1
#         A[t + start_soc_update_pos + 1, t - 1 + start_soc + 1] = -1
#         A[t + start_soc_update_pos + 1, t - 1 + start_d + 1] = 1
#         A[t + start_soc_update_pos + 1, t - 1 + start_c + 1] = -1
#
#         A[t + start_soc_update_neg + 1, t + start_soc + 1] = -1
#         A[t + start_soc_update_neg + 1, t - 1 + start_soc + 1] = 1
#         A[t + start_soc_update_neg + 1, t - 1 + start_d + 1] = -1
#         A[t + start_soc_update_neg + 1, t - 1 + start_c + 1] = 1
#
#     A[t + start_soc_update_neg + 2, t + start_soc] = 1
#     b[t + start_constrain_soc + 2] = soc_0
#
#     A[t + start_soc_update_neg + 3, t + start_soc] = -1
#     b[t + start_constrain_soc + 3] = -soc_0
#
#     return A, b




def train_forecaster_spo(params_dict, features, prices, optimal_schedules, seq=False, model='cvxpy'):
    list_opti_times = []
    list_matrices = []

    if model == 'cvxpy':

        prob, B, _ = spo_plus_erm_cvxpy(params_dict=params_dict, examples=features,
                                        prices=prices,
                                        optimal_schedules=optimal_schedules)

        tic = time.time()

        if seq is False:

            try:
                prob.solve(solver='GUROBI', verbose=True, reoptimize=True)
                matrix = B.value

            except cp.SolverError:
                # handle failure
                A, b = get_full_matrix_problem(params_dict)
                prob_dim = A.shape[1]
                n_features = features.shape[1]

                matrix = np.zeros((prob_dim, n_features))

            print(f"Train time batch: {time.time() - tic} \n")

            return matrix, prob.status

        if seq:

            tolerances = np.logspace(10, -8, 10)
            list_matrices = []
            for tol in tolerances:
                prob.solve(solver=cp.ECOS, abstol=tol, reltol=tol, feastol=tol)
                list_matrices.append(B.value)

            return list_matrices, prob.status

    elif model == 'pyomo':

        #Define model

        prob, B, _ = spo_plus_erm_pyomo(params_dict=params_dict, examples=features,
                                        prices=prices,
                                        optimal_schedules=optimal_schedules)

        #Solve model with callbacks

        A, b = get_full_matrix_problem(params_dict)
        prob_dim = A.shape[1]
        n_features = features.shape[1]

        # Store intermediate solutions of B
        intermediate_solutions = []

        # Define a callback function to access intermediate solutions
        def intermediate_callback(model, alg):
            intermediate_solution = np.zeros((prob_dim, n_features))
            for i in range(prob_dim):
                for j in range(n_features):
                    intermediate_solution[i, j] = pyo.value(model.B[i, j])
                    intermediate_solution[i,j] = model.B[i,j].value
            intermediate_solutions.append(intermediate_solution)

        # Solve the model with an interior point solver and get callbacks for intermediate solutions
        opt = SolverFactory('ipopt')

        # Add the intermediate callback to the solver
        opt.set_callback(intermediate_callback)

        results = opt.solve(model, tee=False)

        print(f"Train time batch: {results.solver.time} \n")

        return intermediate_solutions, str(results.solver.termination_condition)









##### FUNCTIONS FOR OPTIMIZING SCHEDULE #####

import scipy.optimize as so
import copy
import time
import pickle
import os
from tensorflow import keras
import pyomo.environ as pe


def get_models(type_list, read_code_list, la_list, list_model_package, machine):
    models = []
    scalers = []

    if machine == 'vsc':
        loc_models = 'output/trained_models/'
    elif machine == 'local':
        loc_models = 'C:/Users/u0137781/OneDrive - KU Leuven/Imbalance_price_forecast/Python scripts/trained_models/'

    for model_index in range(len(type_list)):
        type = type_list[model_index]
        la = la_list[model_index]
        code = read_code_list[model_index]
        pack = list_model_package[model_index]

        if machine == 'vsc':
            dir_models = loc_models
        elif machine == 'local':
            dir_models = loc_models + type + '/' + 'LA' + str(la) + '/'

        if type == 'OP':
            models.append('NA')
            scalers.append('NA')

        elif type == 'PF':
            models.append('PF')
            scalers.append('NA')

        elif type in ['LM', 'RF']:
            filename = type + '//' + type + '_' + read_code_list[model_index] + '.sav'
            model = pickle.load(open(loc_models + filename, 'rb'))
            models.append(model)

        else:
            if pack == 'tf':
                dir_models = loc_models + type + '//' + 'LA' + str(la) + '//'
                if code == '20220804_20201112_cLoss_opti':
                    model = keras.models.load_model(dir_models + code + '//',
                                                    custom_objects={'loss': cl.custom_loss_stdev_tune(0)})
                elif code == 'full_run20220812_20201112_incl_huber2':
                    model = keras.models.load_model(dir_models + code + '//',
                                                    custom_objects={'loss_fct': cl.custom_loss_Delta_tune(0.25),
                                                                    'loss': cl.custom_loss_Delta_tune(0.25)},
                                                    compile=False)
                else:
                    model = keras.models.load_model(dir_models + code + '//', compile=False)
                models.append(model)
            elif pack[0:5] == 'torch':
                if machine == 'vsc':
                    path_model = dir_models + code + '.pt'
                elif machine == 'local':
                    path_model = loc_models + type + '//' + 'LA' + str(la) + '//pytorch//' + code + '.pt'
                # if type == 'NN':
                # model = T.NeuralNetWithOpti(dict['n_units_1'],dict['n_units_2'],params_dict)

                model = torch.load(path_model)
                model.eval()
                models.append(model)
            elif pack == 'spo_LR':
                if machine == 'vsc':
                    path_model = loc_models + code + '.npy'
                elif machine == 'local':
                    path_model = f"{loc_models + type}//LA{str(la)}//{code}.npy"
                model = np.load(path_model)
                models.append(model)

        dir_scalers = dir_models + code + '_scalers'
        if os.path.isdir(dir_scalers):
            feature_scaler = pickle.load(open(dir_scalers + '//feature_scaler.pkl', 'rb'))
            label_scaler = pickle.load(open(dir_scalers + '//label_scaler.pkl', 'rb'))
            scalers.append([feature_scaler, label_scaler])
        else:
            scalers.append('NA')

    return models, scalers


def optimize_schedule_2(params_dict, opti_settings_dict, imb_price_per_model_test, test_features_dict, list_models,
                        list_model_types):
    # Initialize parameters that are needed globally
    n_models = len(list_models)
    n_qhs = params_dict['n_qhs']
    max_charge = params_dict['max_charge']
    max_discharge = params_dict['max_discharge']
    max_soc = params_dict['max_soc']
    min_soc = params_dict['min_soc']
    lookahead = params_dict['lookahead']
    soc_0_stat = params_dict['soc_0']

    opti_package = opti_settings_dict['opti_package']
    opti_type = opti_settings_dict['opti_type']

    # Define arrays to capture optimization output in the loop
    forward_schedule_all_models = np.zeros((n_models, n_qhs, lookahead))
    forward_soc_all_models = np.zeros((n_models, n_qhs, lookahead))
    optimization_times_all_models = np.zeros((n_models, n_qhs))
    degradation_cost_all_models = np.zeros((n_models, n_qhs))
    count_status_not_optimal = np.zeros(n_models)

    for model_index in range(len(list_models)):
        model_type = list_model_types[model_index]
        count_not_optimal = 0

        if (model_type in ['NN', 'edRNN', 'LR']) or (opti_type == 'exo'):

            bounds = initialize_opti_bounds_scipy(lookahead, min_soc, max_soc, max_discharge, max_charge)

            for qh in range(n_qhs):
                if (qh % 1000) == 0:
                    print(f"starting qh: {qh}")
                if (qh > 0) & params_dict['soc_update']:
                    soc_0 = forward_soc_all_models[model_index, qh - 1, 0]
                else:
                    soc_0 = soc_0_stat

                if opti_type == 'endo':

                    if model_type == 'NN':
                        features_instance = test_features_dict['NN'][qh, :].reshape(1, 294)

                    if model_type == 'edRNN':
                        features_instance = [test_features_dict['edRNN'][0][qh, :, :].reshape(1, 4, 30),
                                             test_features_dict['edRNN'][1][qh, :, :].reshape(1, 10, 29)]

                    forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                    optimization_times_all_models[model_index, qh] = \
                        run_single_optimization_scipy(opti_type=opti_type, bounds=bounds, params_dict=params_dict,
                                                      soc_0=soc_0, model_type=model_type,
                                                      features_instance=features_instance,
                                                      model=list_models[model_index])
                elif opti_type == 'exo':
                    if opti_package == 'scipy':
                        forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                        optimization_times_all_models[model_index, qh] = \
                            run_single_optimization_scipy(opti_type=opti_type, bounds=bounds, params_dict=params_dict,
                                                          soc_0=soc_0, model_type=model_type,
                                                          imb_price_forecasted=imb_price_per_model_test[model_index, qh,
                                                                               :])

                    elif opti_package == 'cvxpy':
                        forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                            optimization_times_all_models[model_index, qh], degradation_cost_all_models[
                            model_index, qh], status = \
                            run_single_optimization_cvxpy(opti_type=opti_type, bounds=bounds, params_dict=params_dict,
                                                          soc_0=soc_0, model_type=model_type,
                                                          imb_price_forecasted=imb_price_per_model_test[model_index, qh,
                                                                               :])
                        if status != 'optimal':
                            count_not_optimal += 1
                    elif opti_package == 'pyomo':
                        forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                            optimization_times_all_models[model_index, qh] = \
                            run_single_optimization_pyomo(params_dict=params_dict,
                                                          soc_0=soc_0,
                                                          pred_imb_price=imb_price_per_model_test[model_index, qh, :])

                    else:
                        sys.exit(f'Opti package {opti_package} does not have a function for optimizing the schedule')

                    count_status_not_optimal[model_index] = count_not_optimal

        elif model_type == 'OP':

            for qh in range(n_qhs):
                print(qh)
                if (qh % 1000) == 0:
                    print(qh)
                if qh > 0:
                    soc_0 = forward_soc_all_models[model_index, qh - 1, 0]

                if opti_type == 'exo':

                    forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                    optimization_times_all_models[model_index, qh] = \
                        run_single_optimization_pyomo(imb_price_per_model_test[model_index, qh, :], params_dict, soc_0)
                elif opti_type == 'endo':
                    if list_models[model_index] == 'NA':
                        features = test_features_dict['vRNN']
                        MO = features[qh, :, 0:20]
                        scenarios_SI = features[qh, :, 20:29]

                        print('Max SI scenario: ' + str(np.max(scenarios_SI)) + '; Min SI scenarios: ' + str(
                            np.min(scenarios_SI)))

                        forward_schedule_all_models[model_index, qh, :], forward_soc_all_models[model_index, qh, :], \
                        optimization_times_all_models[model_index, qh] = \
                            run_single_optimization_pyomo_endo(params_dict, soc_0, MO, scenarios_SI)

    output_dict = {
        'forward_schedule': forward_schedule_all_models,
        'forward_soc': forward_soc_all_models,
        'opti_time': optimization_times_all_models,
        'degr_costs': degradation_cost_all_models,
        'count_not_optimal': count_status_not_optimal
    }

    return output_dict


def initialize_opti_bounds_scipy(lookahead, min_soc, max_soc, max_discharge, max_charge):
    lb_dis = np.zeros(lookahead)
    lb_ch = np.zeros(lookahead)
    lb_soc = np.ones(lookahead) * min_soc
    ub_dis = np.ones(lookahead) * max_discharge
    ub_ch = np.ones(lookahead) * max_charge
    ub_soc = np.ones(lookahead) * max_soc

    lb = np.concatenate((lb_dis, lb_ch, lb_soc), axis=0)
    ub = np.concatenate((ub_dis, ub_ch, ub_soc), axis=0)
    bounds = so.Bounds(lb, ub)

    return bounds


def run_single_optimization_cvxpy(opti_type, bounds, params_dict, soc_0, model_type='OP',
                                  imb_price_forecasted=np.zeros(10), features_instance=np.zeros(10), model='NA'):
    params = copy.deepcopy(params_dict)
    params['soc_0'] = soc_0

    prob, price, net_discharge, soc, degr_cost = opti_problem(params)

    price.value = imb_price_forecasted

    tic = time.time()
    # print('check')
    prob.solve(solver='GUROBI', verbose=False, MIPGap=0.01, TimeLimit=0.5)
    optimization_time = time.time() - tic
    # print(f"optimization time: {optimization_time}")

    forward_schedule = net_discharge.value
    forward_soc = soc.value

    return forward_schedule, forward_soc, optimization_time, degr_cost.value[0], prob.status


def run_single_optimization_scipy(opti_type, bounds, params_dict, soc_0, model_type='OP',
                                  imb_price_forecasted=np.zeros(10), features_instance=np.zeros(10), model='NA'):
    # Initialize parameters that are needed locally

    eff_d = params_dict['eff_d']
    eff_c = params_dict['eff_c']
    ts_len = params_dict['ts_len']
    lookahead = params_dict['lookahead']
    max_discharge = params_dict['max_discharge']
    cyclic_bc = params_dict['cyclic_bc']
    # n_quantiles = params_dict['n_quantiles']

    # Initialize return arrays
    optimized_array = np.zeros(3 * lookahead)
    forward_schedule = np.zeros(lookahead)
    forward_soc = np.zeros(lookahead)

    # Functions for objective and constraints
    def objective_exo(x):

        # Exogenous price formation
        return (-np.dot(np.dot(eff_d, x[0:lookahead]) - np.dot(1 / eff_c, x[lookahead:2 * lookahead]),
                        imb_price_forecasted))

    def objective_endo_nn(x):

        # Endogenous price formation NN10
        features = copy.deepcopy(features_instance)
        for la in range(lookahead):
            for q in range(n_quantiles):
                features[0, la * 29 + q] += (x[la] * eff_d - x[lookahead + la] / eff_c) / ts_len
        price = model.predict(features)
        return (-np.dot(np.dot(eff_d, x[0:lookahead]) - np.dot(1 / eff_c, x[lookahead:2 * lookahead]), price[0, :]))

    def objective_endo_edRNN(x):
        features_fut = copy.deepcopy(features_instance[1])
        for la in range(lookahead):
            for q in range(n_quantiles):
                features_fut[0, la, 20 + q] += (x[la] * eff_d - x[lookahead + la] / eff_c) / ts_len
        price = model.predict([features_instance[0], features_fut])
        return (-np.dot(np.dot(eff_d, x[0:lookahead]) - np.dot(1 / eff_c, x[lookahead:2 * lookahead]), price[0, :]))

    def cons_soc_init(x):
        return x[2 * lookahead] - soc_0 + x[0] - x[lookahead]

    def cons_soc_update(x):
        net_charge = -x[1:lookahead] + x[lookahead + 1:2 * lookahead]
        shifted_soc = x[2 * lookahead:3 * lookahead - 1]
        soc = x[2 * lookahead + 1:3 * lookahead]
        return net_charge + shifted_soc - soc

    def cons_charge_discharge(x):
        return np.ones(lookahead) * max_discharge - x[0:lookahead] - x[lookahead:2 * lookahead]

    def cons_cyclic_bc(x):
        return x[-1] - soc_0

    constraint1 = {'type': 'eq', 'fun': cons_soc_init}
    constraint2 = {'type': 'eq', 'fun': cons_soc_update}
    constraint3 = {'type': 'ineq', 'fun': cons_charge_discharge}
    constraint4 = {'type': 'eq', 'fun': cons_cyclic_bc}
    constraints = [constraint1, constraint2, constraint3]

    if cyclic_bc:
        constraints.append(constraint4)

    guess_1 = np.concatenate((np.zeros(lookahead), np.zeros(lookahead), np.ones(lookahead) * soc_0), axis=0)

    tic = time.perf_counter()
    if opti_type == 'exo':
        res_1 = so.minimize(objective_exo, guess_1, method="SLSQP", tol=0.001, bounds=bounds,
                            constraints=constraints)
    elif model_type == 'NN':
        res_1 = so.minimize(objective_endo_nn, guess_1, method="SLSQP", tol=0.001, bounds=bounds,
                            constraints=constraints)
    elif model_type == 'edRNN':
        res_1 = so.minimize(objective_endo_edRNN, guess_1, method="SLSQP", tol=0.001, bounds=bounds,
                            constraints=constraints)
    toc = time.perf_counter()

    optimized_array[:] = res_1['x']
    for la in range(lookahead):
        net_discharge = res_1['x'][la] * eff_d - res_1['x'][la + lookahead] / eff_c
        forward_schedule[la] = net_discharge
        """
        net_discharge = res_1['x'][la] - res_1['x'][la + lookahead]
        if net_discharge > 0.0001:
            forward_schedule[la] = net_discharge * eff_d
        elif net_discharge < -0.0001:
            forward_schedule[la] = net_discharge / eff_c
        else:
            forward_schedule[la] = 0
        """
    forward_soc[:] = res_1['x'][2 * lookahead:3 * lookahead]
    optimization_time = toc - tic

    return forward_schedule, forward_soc, optimization_time


def build_model_pyomo(params_dict, pred_imb_price, soc_0):
    tic = time.perf_counter()
    # Load model parameters
    eff_d = params_dict['eff_d']
    eff_c = params_dict['eff_c']
    ts_len = params_dict['ts_len']
    lookahead = params_dict['lookahead']
    max_discharge = params_dict['max_discharge']
    max_soc = params_dict['max_soc']
    min_soc = params_dict['min_soc']

    # Define bounds functions
    def _bounds_discharge(model, index_1):
        return (0, max_discharge)

    def _bounds_soc(model, index_1):
        return (min_soc, max_soc)

    # Define model
    model = pe.ConcreteModel()

    # Define sets
    model.T = pe.RangeSet(1, lookahead)

    # Define and initialize variables
    model.discharge = pe.Var(model.T, bounds=_bounds_discharge)
    model.charge = pe.Var(model.T, bounds=_bounds_discharge)
    model.soc = pe.Var(model.T, bounds=_bounds_soc)

    for t in model.T:
        model.discharge[t].value = 0
        model.charge[t].value = 0
        model.soc[t].value = soc_0

    # Define objective
    model.profit = pe.Objective(sense=-1, expr=sum(
        (model.discharge[t] * eff_d - model.charge[t] / eff_c) * pred_imb_price[t - 1] for t in model.T))

    # Define constraints
    model.soc_update = pe.ConstraintList()
    for t in model.T:
        if t == 1:
            model.soc_update.add(model.soc[t] == soc_0 + model.charge[t] - model.discharge[t])
        if t > 1:
            model.soc_update.add(model.soc[t] == model.soc[t - 1] + model.charge[t] - model.discharge[t])

    model.charge_discharge_limit = pe.ConstraintList()
    for t in model.T:
        model.charge_discharge_limit.add(model.charge[t] + model.discharge[t] <= max_discharge)

    return model


def run_single_optimization_pyomo(pred_imb_price, params_dict, soc_0):
    # Initialize return arrays
    eff_d = params_dict['eff_d']
    eff_c = params_dict['eff_c']
    lookahead = params_dict['lookahead']
    forward_schedule = np.zeros(lookahead)
    forward_soc = np.zeros(lookahead)

    tic = time.time()
    model = build_model_pyomo(params_dict, pred_imb_price, soc_0)
    build_time = time.time() - tic

    solver = pe.SolverFactory('gurobi')

    tic = time.time()
    solver.solve(model)
    optimization_time = time.time() - tic

    # Fill and return arrays
    tic = time.time()
    for t in model.T:
        forward_schedule[t - 1] = model.discharge[t]() * eff_d - model.charge[t]() / eff_c
        forward_soc[t - 1] = model.soc[t]()
    post_proc_time = time.time() - tic

    print(f"optimization time: {optimization_time}; build time: {build_time}; post-processing time: {post_proc_time}")

    return forward_schedule, forward_soc, optimization_time


def run_single_optimization_pyomo_endo(params_dict, soc_0, MO, scenarios_SI):
    # Load model parameters
    SI_quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    p = get_probabilities_from_quantiles(SI_quantiles)

    eff_d = params_dict['eff_d']
    eff_c = params_dict['eff_c']
    ts_len = params_dict['ts_len']
    lookahead = params_dict['lookahead']
    max_discharge = params_dict['max_discharge']
    max_soc = params_dict['max_soc']
    min_soc = params_dict['min_soc']
    nb_lvls = 10
    lvl_size = 100 * ts_len
    nb_scen = len(SI_quantiles)

    MO_u = MO[:, nb_lvls:2 * nb_lvls]
    MO_d = MO[:, 0:nb_lvls]

    # Initialize return arrays
    forward_schedule = np.zeros(lookahead)
    forward_soc = np.zeros(lookahead)

    # Define bounds functions
    def _bounds_discharge(model, index_1):
        return (0, max_discharge)

    def _bounds_soc(model, index_1):
        return (min_soc, max_soc)

    def _bounds_activation(model, index_t, index_lvl, index_s):
        return (0, lvl_size)

    def _bounds_imb_price(model, index_t, index_s):
        return (np.min(MO_d), np.max(MO_u))

    ### Build model ###

    # Big-Ms
    M_up_1 = 10000
    M_up_2 = lvl_size
    M_up_3 = lvl_size
    M_up_4 = 10000
    M_down_1 = 10000
    M_down_2 = lvl_size
    M_down_3 = lvl_size
    M_down_4 = 10000

    # Define model
    model = pe.ConcreteModel()

    # Define sets
    model.T = pe.RangeSet(1, lookahead)
    model.LU = pe.RangeSet(1, nb_lvls)
    model.LD = pe.RangeSet(1, nb_lvls)
    model.S = pe.RangeSet(1, nb_scen)

    # Define and initialize variables
    model.discharge = pe.Var(model.T, bounds=_bounds_discharge)
    model.charge = pe.Var(model.T, bounds=_bounds_discharge)
    model.soc = pe.Var(model.T, bounds=_bounds_soc)

    model.imb_price = pe.Var(model.T, model.S, bounds=_bounds_imb_price)

    model.act_ene_u = pe.Var(model.T, model.LU, model.S, bounds=_bounds_activation)
    model.act_ene_d = pe.Var(model.T, model.LD, model.S, bounds=_bounds_activation)

    model.mu_per_level_u = pe.Var(model.T, model.LU, model.S)
    model.mu_per_level_d = pe.Var(model.T, model.LD, model.S)

    model.z_up_1 = pe.Var(model.T, model.LU, model.S, domain=pe.Boolean)
    model.z_up_2 = pe.Var(model.T, model.LU, model.S, domain=pe.Boolean)

    model.z_down_1 = pe.Var(model.T, model.LU, model.S, domain=pe.Boolean)
    model.z_down_2 = pe.Var(model.T, model.LU, model.S, domain=pe.Boolean)

    for t in model.T:
        model.discharge[t].value = 0
        model.charge[t].value = 0
        model.soc[t].value = soc_0

    # Define objective
    model.profit = pe.Objective(sense=-1,
                                expr=sum(p[s - 1] * (
                                        - sum(model.imb_price[t, s] * ts_len * scenarios_SI[t - 1, s - 1] for t in
                                              model.T)
                                        - (lvl_size * sum(
                                    model.mu_per_level_u[t, l, s] for t in model.T for l in model.LU) + sum(
                                    MO_u[t - 1, l - 1] * model.act_ene_u[t, l, s] for t in model.T for l in model.LU))
                                        + (-lvl_size * sum(
                                    model.mu_per_level_d[t, l, s] for t in model.T for l in model.LD) + sum(
                                    MO_d[t - 1, l - 1] * model.act_ene_d[t, l, s] for t in model.T for l in model.LD))
                                )
                                         for s in model.S)
                                )

    # Define constraints

    # Upper level constraints
    model.soc_update = pe.ConstraintList()
    for t in model.T:
        if t == 1:
            model.soc_update.add(model.soc[t] == soc_0 + model.charge[t] - model.discharge[t])
        if t > 1:
            model.soc_update.add(model.soc[t] == model.soc[t - 1] + model.charge[t] - model.discharge[t])

    model.charge_discharge_limit = pe.ConstraintList()
    for t in model.T:
        model.charge_discharge_limit.add(model.charge[t] + model.discharge[t] <= max_discharge)

    # Lower level constraints

    model.con_balance = pe.ConstraintList()
    for t in model.T:
        for s in model.S:
            model.con_balance.add(sum(model.act_ene_u[t, l, s] - model.act_ene_d[t, l, s] for l in model.LU) + (
                        model.discharge[t] * eff_d - model.charge[t] / eff_c) + ts_len * scenarios_SI[
                                      t - 1, s - 1] == 0)

    model.con_c1 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_c1.add(
                    MO_u[t - 1, l - 1] - model.imb_price[t, s] + model.mu_per_level_u[t, l, s] <= M_up_1 * (
                                1 - model.z_up_1[t, l, s]))

    model.con_c2 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_c2.add(MO_u[t - 1, l - 1] - model.imb_price[t, s] + model.mu_per_level_u[t, l, s] >= 0)

    model.con_d = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_d.add(model.act_ene_u[t, l, s] <= M_up_2 * model.z_up_1[t, l, s])

    model.con_e1 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_e1.add(
                    -MO_d[t - 1, l - 1] + model.imb_price[t, s] + model.mu_per_level_d[t, l, s] <= M_down_1 * (
                                1 - model.z_down_1[t, l, s]))

    model.con_e2 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_e2.add(-MO_d[t - 1, l - 1] + model.imb_price[t, s] + model.mu_per_level_d[t, l, s] >= 0)

    model.con_f = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_f.add(model.act_ene_d[t, l, s] <= M_down_2 * model.z_down_1[t, l, s])

    model.con_h1 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_h1.add(lvl_size - model.act_ene_u[t, l, s] <= M_up_3 * model.z_up_2[t, l, s])

    model.con_h2 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_h2.add(lvl_size - model.act_ene_u[t, l, s] >= 0)

    model.con_i = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_i.add(model.mu_per_level_u[t, l, s] <= M_up_4 * (1 - model.z_up_2[t, l, s]))

    model.con_j1 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_j1.add(lvl_size - model.act_ene_d[t, l, s] <= M_down_3 * model.z_down_2[t, l, s])

    model.con_j2 = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_j2.add(lvl_size - model.act_ene_d[t, l, s] >= 0)

    model.con_k = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_k.add(model.mu_per_level_d[t, l, s] <= M_down_4 * (1 - model.z_down_2[t, l, s]))

    model.con_mu_positive = pe.ConstraintList()
    for t in model.T:
        for l in model.LU:
            for s in model.S:
                model.con_mu_positive.add(model.mu_per_level_u[t, l, s] >= 0)
                model.con_mu_positive.add(model.mu_per_level_d[t, l, s] >= 0)

    # Choose solver
    # solver = pe.SolverFactory(solvername, executable=solverpath_exe)
    solver = pe.SolverFactory('gurobi')

    tic = time.perf_counter()
    solver.solve(model)
    toc = time.perf_counter()
    # print('optimizaton time: '+str(toc-tic))

    # Fill and return arrays
    for t in model.T:
        forward_schedule[t - 1] = model.discharge[t]() * eff_d - model.charge[t]() / eff_c
        forward_soc[t - 1] = model.soc[t]()
        optimization_time = toc - tic

    print(forward_schedule)

    return forward_schedule, forward_soc, optimization_time


##### Post-optimization calculations #####

def calc_profit_per_qh(discharge_schedules, imb_price):
    n_models = discharge_schedules.shape[0]
    n_qhs = discharge_schedules.shape[1]
    profit_per_qh = np.zeros((n_models, n_qhs))

    for model in range(n_models):
        for qh in range(n_qhs):
            profit_per_qh[model, qh] = discharge_schedules[model, qh, 0] * imb_price[qh, 0]

    return profit_per_qh


def calc_sum_profits(profits_per_qh):
    n_models = profits_per_qh.shape[0]
    sum_profit = np.zeros(n_models)
    for model in range(n_models):
        sum_profit[model] = np.sum(profits_per_qh[model, :])

    return sum_profit


def calculate_error(calculated_imb_price, labels):
    error = np.zeros(calculated_imb_price.shape)

    for model in range(error.shape[0]):
        error[model, :, :] = calculated_imb_price[model, :, :] - labels

    abs_error = np.abs(error)
    squared_error = np.square(error)
    third_error = np.power(abs_error, 3)
    return error, abs_error, np.mean(error, axis=1), np.mean(abs_error, axis=1), np.mean(squared_error,
                                                                                         axis=1), np.mean(third_error,
                                                                                                          axis=1)


def get_inter_qh_delta(array):
    if len(array.shape) == 2:
        inter_qh_delta = calc_inter_qh_delta_2d(array)

    elif len(array.shape) == 3:
        n_models = array.shape[0]
        n_instances = array.shape[1]
        lookahead = array.shape[2]

        inter_qh_delta = np.zeros((n_models, n_instances, lookahead - 1))

        for model in range(n_models):
            inter_qh_delta[model, :, :] = calc_inter_qh_delta_2d(array[model, :, :])


    else:
        print('Input array should be 2d or 3d')
        sys.exit()

    return inter_qh_delta


def calc_inter_qh_delta_2d(array_2d):
    n_instances = array_2d.shape[0]
    lookahead = array_2d.shape[1]

    if lookahead > 1:
        inter_qh_delta_2d = np.zeros((n_instances, lookahead - 1))

        for inst in range(n_instances):
            for la in range(lookahead - 1):
                inter_qh_delta_2d[inst, la] = array_2d[inst, la + 1] - array_2d[inst, la]

    else:
        print('Cannot calculate delta if Lookahead <= 1')
        sys.exit()

    return inter_qh_delta_2d


def get_delta_errors(predicted_deltas, actual_deltas):
    n_models = predicted_deltas.shape[0]
    n_instances = predicted_deltas.shape[1]
    delta_lookahead = predicted_deltas.shape[2]

    delta_errors = np.zeros((n_models, n_instances, delta_lookahead))

    for model in range(n_models):
        delta_errors[model, :, :] = predicted_deltas[model, :, :] - actual_deltas

    avg_delta_error_la = np.mean(delta_errors, axis=1)
    avg_abs_delta_error_la = np.mean(np.abs(delta_errors), axis=1)

    return delta_errors, avg_delta_error_la, avg_abs_delta_error_la
