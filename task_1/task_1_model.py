# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from torch_module.layers.core import DNN, PredictionLayer
from torch_module.models.basemodel import BaseModel


class MLP(BaseModel):
    def __init__(self, input_dim, hidden_units, activation='relu', device='cpu', use_bn=True):
        super(MLP, self).__init__(device=device)
        self.mlp = DNN(input_dim, hidden_units, activation=activation, device=device, use_bn=use_bn)
        self.predict_header = PredictionLayer(task='binary', use_bias=False)

        self.to(device)

    def forward(self, input):
        return self.predict_header(self.mlp(input))

