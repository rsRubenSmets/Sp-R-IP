import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable,Function
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from functions_support import opti_problem

class Loss_profit(nn.Module):

    def __init__(self):
        super(Loss_profit, self).__init__()

    def forward(self, output, prices):
        schedules = output
        profit = torch.sum(torch.mul(schedules, prices))

        return -profit

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

