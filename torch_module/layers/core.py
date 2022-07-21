import math
import torch
import torch.nn.functional as F
from torch import nn as nn

from .activation import activation_layer


class DNN(nn.Module):

    def __init__(self, input_dim, hidden_units, activation='relu', dropout_rate=0, seed=2022, l2_reg=0,
                 use_bn=False, init_std=0.0001, device='cpu', init_weight='normal'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.init_weight = init_weight
        if len(hidden_units) == 0:
            raise ValueError("Hidden_units is empty!")
        hidden_units = [input_dim] + list(hidden_units)

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)]
            )

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation) for _ in range(len(hidden_units) - 1)]
        )

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                if isinstance(self.init_weight, str):
                    if self.init_weight.lower() == 'normal':
                        nn.init.normal_(param, mean=0, std=init_std)
                    elif self.init_weight.lower() == 'kaiming_uniform':
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    elif self.init_weight.lower() == 'xavier_uniform':
                        nn.init.xavier_uniform_(param)
                    else:
                        raise NotImplementedError("{} param init method is Not Implemented!".format(self.init_weight))

        self.to(device)

    def forward(self, inputs):
        fc_input = inputs
        for i in range(len(self.linear_layers)):
            fc = self.linear_layers[i](fc_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            fc_input = fc
        return fc_input


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        output = x
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(output)
        elif self.task == 'multiclass':
            output = torch.softmax(output, dim=-1)
        return output
