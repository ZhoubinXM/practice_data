import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data

from ..utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error, accuracy_score
from tensorflow.python.keras.callbacks import CallbackList, History


class BaseModel(nn.Module):
    def __init__(self, device='cpu'):
        super(BaseModel, self).__init__()
        self.device = device
        self.history = History()

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            log_dir=None, validation_data=None, shuffle=True, callbacks=None):
        # Tensorboard
        self.tensorboard_log_dir = log_dir
        if self.tensorboard_log_dir:
            self.writer = SummaryWriter(self.tensorboard_log_dir)

        if isinstance(x, pd.DataFrame):
            x = [x[feature] for feature in list(x.columns)]

        do_validation = False
        if validation_data:
            do_validation = True
            raise ValueError("No Validation data input.")
        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at), slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at), slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        y = np.expand_dims(y, axis=1)
        train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.tensorboard_log_dir:
            self.writer.add_graph(model, torch.randn((1, torch.from_numpy(np.concatenate(x, axis=-1))[0].shape[0])))

        train_loader = Data.DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # callback method
        callbacks = (callbacks or []) + [self.history]
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        callbacks.model.stop_training = False

        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            start_time = time.time()
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for i, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()

                        y_pre = model(x).squeeze()

                        optim.zero_grad()
                        loss = loss_func(y_pre, y.squeeze(), reduction='sum')
                        reg_loss = 0
                        total_loss = loss + reg_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()
                        loss.backward()
                        optim.step()

                        if self.tensorboard_log_dir:
                            self.writer.add_scalar('loss', loss, (epoch+1)*steps_per_epoch+i)

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                train_result[name].append(metric_fun(
                                    y.cpu().data.numpy(), y_pre.cpu().data.numpy().astype("float64")))
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # Add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(
                    epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            # torch.save(model.state_dict(), './logs/test.pth')
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.tensorboard_log_dir:
                if epoch_logs:
                    train_logs = {k: v for k, v in epoch_logs.items() if not k.startswith('val_')}
                    val_logs = {k: v for k, v in epoch_logs.items() if k.startswith('val')}
                    self.writer.add_scalars('train_log', train_logs, epoch)
                    self.writer.add_scalars('val_logs', val_logs, epoch)
            if self.stop_training:
                break
        callbacks.on_train_end()
        if self.tensorboard_log_dir:
            self.writer.close()

        return self.history

    def evaluate(self, x, y, batch_size):
        val_pre = self.predict(x, batch_size)
        val_res = {}
        for name, metric_func in self.metrics.items():
            val_res[name] = metric_func(y, val_pre)
        return val_res

    def predict(self, x, batch_size):
        model = self.eval()
        if isinstance(x, pd.DataFrame):
            x = [x[feature] for feature in list(x.columns)]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        test_loader = Data.DataLoader(dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def compile(self, optimizer, loss=None, metrics=None):
        self.metrics_names = ['loss']
        self.optim = self._get_optim(optimizer)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer):
        lr = 0.01
        if isinstance(optimizer, list):
            if len(optimizer) == 2:
                lr = optimizer[1]
                optimizer = optimizer[0]
            elif len(optimizer) == 1:
                optimizer = optimizer[0]
            else:
                raise ValueError("optimizer must is a list with [optimizer(str), lr(float)] or [optimizer(str)]")
        if isinstance(optimizer, str):
            if optimizer.lower() == 'sgd':
                optim = torch.optim.SGD(self.parameters(), lr=lr)
            elif optimizer.lower() == 'adam':
                optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.0002)
            elif optimizer.lower() == 'adagrad':
                optim = torch.optim.Adagrad(self.parameters(), lr=lr)
            elif optimizer.lower() == 'rmsprop':
                optim = torch.optim.RMSprop(self.parameters(), lr=lr)
            else:
                raise NotImplementedError("{} is NOTImplemented!".format(optimizer))
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss_type):
        if isinstance(loss_type, str):
            if loss_type.lower() == 'binary_cross_entropy':
                loss_func = F.binary_cross_entropy
            elif loss_type.lower() == 'mse':
                loss_func = F.mse_loss
            elif loss_type.lower() == 'mae':
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss_type
        return loss_func

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if isinstance(metrics, list) or metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = self._accuracy_score
                self.metrics_names.append(metric)
        else:
            raise ValueError("metrics must is list type, like ['auc'...]")
        return metrics_

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true, y_pred, eps, normalize, sample_weight, labels)

    @staticmethod
    def _accuracy_score(y_true, y_pred):
        return accuracy_score(y_true, np.where(y_pred > 0.5, 1, 0))
