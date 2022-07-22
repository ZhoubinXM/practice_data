import torch
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorboardX import SummaryWriter


class ModelCheckpoint(ModelCheckpoint):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            try:
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
            except KeyError as e:
                raise KeyError('Failed to format this callback filepath: "{}". '
                               'Reason: {}'.format(self.filepath, e))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    print('Can save best model only with %s available, skipping.' % self.monitor)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1, self.monitor, self.best,
                                                           current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            torch.save(self.model.state_dict(), filepath)
                        else:
                            torch.save(self.model, filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save(self.model.state_dict(), filepath)
                else:
                    torch.save(self.model, filepath)


class TensorBoard(Callback):
    def __init__(self,
                 log_dir='logs',
                 histogram_freq=0,
                 write_graph=None,
                 dummy_input=None,
                 write_images=False,
                 write_steps_per_second=False,
                 update_freq='epoch',
                 embeddings_freq=0):
        super(TensorBoard, self).__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.dummy_input = dummy_input
        self.write_images = write_images
        self.write_steps_per_second = write_steps_per_second
        self.update_freq = update_freq
        self.embedding_freq = embeddings_freq

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_train_begin(self, logs=None):
        if self.write_graph:
            if self.dummy_input:
                try:
                    self.writer.add_graph(self.model, self.dummy_input)
                except Exception as e:
                    raise ValueError("Error when viz Model graph.\n"
                                     "Error: {}".format(e))

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            train_logs = {k: v for k, v in logs.items() if not k.startwith('val_')}
            val_logs = {k: v for k, v in logs.items() if k.startwith('val_')}
            self.writer.add_scalars('train_log', train_logs)
            self.writer.add_scalars('val_log', val_logs)

    def on_train_end(self, logs=None):
        return self.writer.close()





