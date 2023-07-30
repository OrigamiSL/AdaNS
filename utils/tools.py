import numpy as np
import torch
import torch.nn.functional as F
from statsmodels.tsa.filters.hp_filter import hpfilter


def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        args.learning_rate = args.learning_rate * 0.5
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def remove_trend(x, lmb):
    _, trend = hpfilter(x, lmb)
    x_detrend = x - trend
    return x_detrend


def Psi(x, c):
    return np.sign(x) * np.minimum(np.abs(x), c)


def MAD(x):
    return np.mean(np.abs(x - np.mean(x)))


def reomve_outlier(x, c):
    mu = np.median(x)
    s = MAD(x)
    return Psi((x - mu)/s, c)


def preprocess(x, lmb, c):
    y = np.zeros_like(x)
    for i in range(x.shape[1]):
        y[:, i] = remove_trend(x[:, i], lmb)
        y[:, i] = reomve_outlier(y[:, i], c)
    return y


def preprocess_m4(x, lmb, c):
    y = np.zeros_like(x)
    for i in range(x.shape[0]):
        y[i, :] = remove_trend(x[i, :], lmb)
        y[i, :] = reomve_outlier(y[i, :], c)
    return y


def sorted_index_topk(x, k):
    k_index = []
    for _ in range(int(k)):
        cur_index = np.argmax(x)
        k_index.append(cur_index)
        x[cur_index] = -1
    return sorted(k_index, reverse=True)


def SMAPE_loss(pred, true):
    divide = torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true))
    divide[divide != divide] = .0
    divide[divide == np.inf] = .0
    smape = 200 * torch.mean(divide)
    return smape
