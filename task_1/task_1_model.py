# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from torch_module.layers.core import DNN, PredictionLayer
from torch_module.models.basemodel import BaseModel


class MLP(BaseModel):
    def __init__(self, input_dim, hidden_units, activation='relu', device='cpu', use_bn=True):
        super(MLP, self).__init__(device=device)
        self.mlp = DNN(input_dim, hidden_units[:-1], activation=activation, device=device, use_bn=use_bn)
        self.linear = nn.Linear(hidden_units[-2], hidden_units[-1])
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.predict_header = PredictionLayer(task='binary', use_bias=False)

        self.to(device)

    def forward(self, input):
        return self.predict_header(self.linear(self.mlp(input)))

# class MLP(BaseModel):
#     def __init__(self, input_dim, embedding_size, hidden_size, output_dim):
#         super(MLP, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.embedding_size = embedding_size
#
#         self.hidden1 = torch.nn.Linear(self.input_dim, self.embedding_size)
#         torch.nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
#         self.act1 = torch.nn.ReLU()
#
#         self.hidden2 = torch.nn.Linear(self.embedding_size, hidden_size)
#         torch.nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
#         self.act2 = torch.nn.ReLU()
#
#         self.hidden3 = torch.nn.Linear(hidden_size, 1)
#         torch.nn.init.xavier_uniform_(self.hidden3.weight)
#         self.act3 = torch.nn.Sigmoid()
#
#     def forward(self, input_tensor):
#         return self.act3(self.hidden3(self.act2(self.hidden2(self.act1(self.hidden1(input_tensor))))))
