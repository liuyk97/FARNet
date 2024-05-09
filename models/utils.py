# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import datetime
import logging
import os

import torch
from torch.nn import SyncBatchNorm
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score):

        # score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.score = score


def distribute(net, distribution=True, local_rank=None):
    if distribution:
        net = net.cuda(local_rank)
        net = SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[local_rank],
                                      output_device=local_rank,
                                      find_unused_parameters=True,
                                      broadcast_buffers=False)
    else:
        net = net.cuda()
    return net


def get_logger(logdir, Note):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, 'run_{}_{}.log'.format(ts, Note))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr


def get_scheduler(optimizer, train_iters=90000):
    scheduler = PolynomialLR(optimizer, max_iter=train_iters, gamma=0.9)

    return scheduler


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1,
                 gamma=0.9, last_epoch=-1):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
        #     return [base_lr for base_lr in self.base_lrs]
        # else:
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        factor = max(factor, 0)
        return [base_lr * factor for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, scheduler, mode='linear',
                 warmup_iters=100, gamma=0.2, last_epoch=-1):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == 'linear':
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == 'constant':
                factor = self.gamma
            else:
                raise KeyError('WarmUp type {} not implemented'.format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs


def freeze_bn(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d) \
            or isinstance(m, nn.BatchNorm2d):
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False
