import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable,Function
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from functions_support import opti_problem

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

