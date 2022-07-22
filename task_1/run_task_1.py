# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch

from torch_module.utils import *
from torch_module.layers.core import *
from data_process import data_loader, submit_result
from task_1_model import MLP
from torch_module.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping

if __name__ == "__main__":
    model_name = './model_checkpoints/task_1_test_model.ckpt'
    logs_name = './logs/20220722'
    train_data, train_label, test_data, test_label = data_loader()
    model = MLP(10, [20, 20, 8, 1], device='cpu')
    print(model)

    # callback func implement
    model_checkpoint = ModelCheckpoint(filepath=model_name, monitor='val_acc',
                                       verbose=1, save_best_only=True, save_weights_only=True, mode='max', period=1)
    # model_early_stopping = EarlyStopping(monitor='val_acc', patience=50, verbose=1, mode='max')

    model.compile(['adam', 0.01], "binary_cross_entropy", metrics=['mse', 'auc', 'acc'])
    model.fit(train_data, [train_label], epochs=1000, verbose=2, validation_split=0.2, log_dir=logs_name,
              callbacks=[model_checkpoint
                         # model_early_stopping
                         ])

    model.load_state_dict(torch.load(model_name))
    pre_ans = model.predict(test_data, batch_size=1000)
    pre_ans = np.where(pre_ans > 0.5, 1, 0)
    submit_result(pre_ans, test_label)


