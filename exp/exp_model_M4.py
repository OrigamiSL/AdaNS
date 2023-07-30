from data.data_loader import Dataset_m4
from exp.exp_basic import Exp_Basic
from AdaNS.model import AdaNS

from utils.tools import EarlyStopping, adjust_learning_rate, SMAPE_loss
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings

warnings.filterwarnings('ignore')


class Exp_Model(Exp_Basic):
    def __init__(self, args):
        super(Exp_Model, self).__init__(args)

    def _build_model(self):
        model = AdaNS(
            self.args.input_len,
            self.args.piece_len,
            self.args.pred_len,
            self.args.attn_pieces,
            self.args.attn_nums1,
            self.args.attn_nums2,
            self.args.n_heads,
            self.args.d_model,
            self.args.Sample_strategy,
            self.args.dropout,
        ).float()
        total_num = sum(p.numel() for p in model.parameters())
        print("Total parameters: {0}MB".format(total_num / 1024 / 1024))
        return model

    def _get_data(self, flag, ins_group, predict=False):
        args = self.args
        data_set = None

        data_dict = {
            'M4_Yearly': Dataset_m4,
            'M4_Quarterly': Dataset_m4,
            'M4_Monthly': Dataset_m4,
            'M4_Weekly': Dataset_m4,
            'M4_Daily': Dataset_m4,
            'M4_Hourly': Dataset_m4,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            size=[args.input_len, args.pred_len],
            ins_group=ins_group,
            freq=args.freq,
            predict=predict
        )
        print(flag, len(data_set))
        if len(data_set) < 10 * args.batch_size:
            batch_size = 1
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def vali(self, vali_data=None, vali_loader=None):
        self.model.eval()
        total_loss = []
        criterion = SMAPE_loss
        with torch.no_grad():
            for i, batch_x in enumerate(vali_loader):
                batch_x = batch_x.unsqueeze(-1)
                pred, true = self._process_one_batch(
                    batch_x)
                loss = criterion(pred, true).detach().cpu().numpy()
                total_loss.append(loss)

            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, ins_group=None):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()

        train_data, train_loader = self._get_data(flag='train', ins_group=ins_group)
        vali_data, vali_loader = self._get_data(flag='val', ins_group=ins_group)
        test_data, test_loader = self._get_data(flag='test', ins_group=ins_group)

        time_now = time.time()
        train_steps = len(train_loader)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                model_optim.zero_grad()
                iter_count += 1
                batch_x = batch_x.unsqueeze(-1)
                pred, true = self._process_one_batch(
                    batch_x)
                loss = SMAPE_loss(pred, true)
                
                loss.backward(torch.ones_like(loss))
                model_optim.step()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1,
                                                                            torch.mean(loss).item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Pred_len: {0}| Epoch: {1}, Steps: {2} | Vali Loss: {3:.7f} Test Loss: {4:.7f}".
                  format(self.args.pred_len, epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, ins_group=None, group_index=0, load=False, save_loss=True, load_tuned=False):
        if load_tuned:
            path = os.path.join(self.args.tuned_checkpoints, self.args.data, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        elif load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()

        test_data, test_loader = self._get_data(flag='test', ins_group=ins_group, predict=True)
        time_now = time.time()
        if save_loss:
            preds = []
            trues = []
            naive_data = []
            input_data = []

            with torch.no_grad():
                for i, (batch_x, naive_x) in enumerate(test_loader):
                    naive_x = naive_x.unsqueeze(-1).detach().cpu().numpy()
                    batch_x = batch_x.unsqueeze(-1)
                    pred, true = self._process_one_batch(
                        batch_x)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(true)
                    naive_data.append(naive_x[:, :self.args.pred_len])
                    input_data.append(batch_x.detach().cpu().numpy())
            print("inference time: {}".format(time.time() - time_now))
            preds = np.stack(preds)
            trues = np.stack(trues)
            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            input_data = np.stack(input_data)
            naive_data = np.stack(naive_data)
            input_data = input_data.reshape(-1, input_data.shape[-2], input_data.shape[-1])
            naive_data = naive_data.reshape(-1, naive_data.shape[-2], naive_data.shape[-1])

            smape, owa = metric(preds, trues, naive_data, input_data, self.args.frequency)
            print('|{}_{}|pred_len{}|smape:{}, owa:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, smape, owa) + '\n')
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            np.save(folder_path + f'metrics.npy', np.array([smape, owa]))
            np.save(folder_path + f'pred.npy', preds)
            np.save(folder_path + f'true.npy', trues)

        else:
            smape_list = []
            owa_list = []
            with torch.no_grad():
                for i, (batch_x, naive_x) in enumerate(test_loader):
                    naive_x = naive_x.unsqueeze(-1).detach().cpu().numpy()
                    batch_x = batch_x.unsqueeze(-1)

                    pred, true = self._process_one_batch(
                        batch_x)
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    input_x = batch_x.detach().cpu().numpy()
                    t_smape, t_owa = metric(pred, true, naive_x, input_x, self.args.frequency)
                    smape_list.append(t_smape)
                    owa_list.append(t_owa)

            print("inference time: {}".format(time.time() - time_now))

            smape = np.average(smape_list)
            owa = np.average(owa_list)
            print('|{}_{}|pred_len{}|smape:{}, owa:{}'.
                  format(self.args.data, self.args.features, self.args.pred_len, smape, owa) + '\n')

        path = './result_m4.log'
        if ins_group is None:
            with open(path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                f.write('|{}_{}|pred_len{}|smape:{}, owa:{}'.
                        format(self.args.data, self.args.features, self.args.pred_len, smape,
                               owa) + '\n')
                f.flush()
                f.close()
        else:
            with open(path, "a") as f:
                f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
                f.write('group_{}|{}_{}|pred_len{}|smape:{}, owa:{}'.
                        format(group_index, self.args.data, self.args.features, self.args.pred_len
                               , smape, owa) + '\n')
                f.flush()
                f.close()

        if not save_loss:
            dir_path = os.path.join(self.args.checkpoints, setting)
            check_path = dir_path + '/' + 'checkpoint.pth'
            if os.path.exists(check_path):
                os.remove(check_path)
                os.removedirs(dir_path)

        return smape, owa

    def _process_one_batch(self, batch_x):
        batch_x = batch_x.float().to(self.device)

        input_seq = batch_x[:, :self.args.input_len, :]
        outputs = self.model(input_seq)
        batch_y = batch_x[:, -self.args.pred_len:, :].to(self.device)

        return outputs, batch_y
