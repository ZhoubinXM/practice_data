# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch

from torch_module.utils import *
from torch_module.layers.core import *
from data_process import data_loader, submit_result
from task_1_model import MLP

if __name__ == "__main__":
    train_data, train_label, test_data, test_label = data_loader()
    # train_data = pd2dict(train_data)
    model = MLP(10, [50, 20, 8, 1], device='cpu')
    model.compile(['adam', 0.01], "binary_cross_entropy", metrics=['mse', 'auc', 'acc'])
    model.fit(train_data, [train_label], epochs=100, verbose=2, validation_split=0.2)
    pre_ans = model.predict(test_data, batch_size=1000)
    pre_ans = np.where(pre_ans > 0.5, 1, 0)
    submit_result(pre_ans, test_label)


