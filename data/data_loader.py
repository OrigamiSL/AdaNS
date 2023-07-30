import os
import math, random
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.tools import *
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', var_group=None, w_random=False, factor=5):
        # size [input_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.var_group = var_group
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.input_len, 12 * 30 * 24 + 4 * 30 * 24 - self.input_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values
        if self.var_group is not None:
            df_value = df_value[:, self.var_group].reshape(df_value.shape[0], -1)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.sample_train = train_data
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train_sample(self, sample_length):
        return self.sample_train[-sample_length:, :]


class Dataset_ETT_min(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv',
                 target='OT', var_group=None, w_random=False, factor=5):
        # size [input_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.var_group = var_group
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.input_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.input_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values
        if self.var_group is not None:
            df_value = df_value[:, self.var_group].reshape(df_value.shape[0], -1)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.sample_train = train_data
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)

        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train_sample(self, sample_length):
        return self.sample_train[-sample_length:, :]


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='ECL.csv', target='MT_321'
                 , var_group=None, w_random=False, factor=5):
        # size [input_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.data_path = data_path
        self.features = features
        self.target = target
        self.root_path = root_path
        self.data_path = data_path
        self.var_group = var_group
        self.w_random = w_random
        self.factor = factor
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + [self.target] + cols]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.input_len, len(df_raw) - num_test - self.input_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_value = df_data.values
        if self.var_group is not None:
            df_value = df_value[:, self.var_group].reshape(df_value.shape[0], -1)

        # data standardization
        train_data = df_value[border1s[0]:border2s[0]]
        self.sample_train = train_data
        self.scaler.fit(train_data)
        data = self.scaler.transform(df_value)
        self.data_x = data[border1:border2]

    def __getitem__(self, index):
        r_begin = index
        r_end = r_begin + self.input_len + self.pred_len
        if self.w_random and self.set_type == 0:
            variables = list(range(self.data_x.shape[1]))
            sample_var = random.sample(variables, self.factor * int(math.log2(self.data_x.shape[1])))
            seq_x = self.data_x[r_begin:r_end, np.array(sample_var).reshape(-1)]
        else:
            seq_x = self.data_x[r_begin:r_end]
        return seq_x

    def __len__(self):
        return len(self.data_x) - self.input_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train_sample(self, sample_length):
        return self.sample_train[-sample_length:, :]


class Dataset_m4(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 ins_group=None, freq='Daily', predict=False,
                 sample_length=0, sample=False):
        # size [input_len, pred_len]
        # info
        self.input_len = size[0]
        self.pred_len = size[1]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.freq = freq
        self.root_path = root_path
        self.instance_group = ins_group
        self.predict = predict
        self.sample_length = sample_length
        self.sample = sample
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path + 'Dataset/Train/', self.freq + '-train.csv'))
        df_raw.set_index(df_raw.columns[0], inplace=True)
        if self.set_type == 2:
            df_target = pd.read_csv(os.path.join(self.root_path + 'Dataset/Test/', self.freq + '-test.csv'))
            df_target.set_index(df_target.columns[0], inplace=True)
            target_data = df_target.values
            df_info = pd.read_csv(os.path.join(self.root_path, 'Dataset/M4-info.csv'))
            df_info.set_index(df_info.columns[0], inplace=True)
            self.info = df_info['SP']
            df_naive = pd.read_csv(os.path.join(self.root_path, 'Point Forecasts/submission-Naive2.csv'))
            df_naive.set_index(df_naive.columns[0], inplace=True)
            naive_data = df_naive.values
            naive_data = naive_data[self.info == self.freq]
        data_x = df_raw.values

        self.data_x = []
        self.target_data = []
        self.naive_data = []
        self.train_lens = []
        train_lens = []
        self.vali_lens = []
        self.min_lens = []
        if self.sample:
            self.sample_data = []
        for index in range(len(data_x)):
            data_current = data_x[index]
            data_current = data_current[~np.isnan(data_current)]
            self.min_lens.append(len(data_current) - self.pred_len)
            if self.sample:
                if self.sample_length > len(data_current):
                    zeros = np.zeros(self.sample_length)
                    self.sample_data.append(zeros)
                else:
                    self.sample_data.append(data_current[-self.sample_length:])
            data_len = data_current.shape[0] - self.input_len - self.pred_len
            if self.set_type == 0:  # train
                if data_len * 0.25 >= 1:
                    self.data_x.append(data_current[0:int(data_len * 0.75) + self.input_len + self.pred_len])
                    self.train_lens.append(int(data_len * 0.75))
                    train_lens.append(data_current.shape[0])
                else:
                    if int(data_len) < 2:
                        self.data_x.append(data_current[:self.input_len + self.pred_len])
                        self.train_lens.append(1)
                    else:
                        self.data_x.append(data_current[0:int(data_len) + self.input_len + self.pred_len - 1])
                        self.train_lens.append(int(data_len) - 1)
            elif self.set_type == 1:  # val
                if data_len * 0.25 >= 1:
                    self.data_x.append(data_current[int(data_len * 0.75):])
                    self.vali_lens.append(data_len - int(data_len * 0.75))
                else:
                    self.data_x.append(data_current[-(self.input_len + self.pred_len):])
                    self.vali_lens.append(1)
            if self.set_type == 2:
                zeros = np.zeros(self.input_len)
                data_current = np.concatenate([zeros, data_current], axis=-1)
                self.data_x.append(data_current[-self.input_len:])
                target_current = target_data[index]
                target_current = target_current[~np.isnan(target_current)]
                self.target_data.append(target_current)
                naive_current = naive_data[index]
                naive_current = naive_current[~np.isnan(naive_current)]
                self.naive_data.append(naive_current)

        if self.instance_group is not None:
            self.data_x = [self.data_x[var] for var in self.instance_group]
            zeros = np.zeros(1)
            if self.set_type == 0:
                self.train_lens = np.array(self.train_lens)
                self.train_lens = np.cumsum(self.train_lens[self.instance_group])
                self.train_lens = np.concatenate([zeros, self.train_lens], axis=0)
            elif self.set_type == 1:
                self.vali_lens = np.array(self.vali_lens)
                self.vali_lens = np.cumsum(self.vali_lens[self.instance_group])
                self.vali_lens = np.concatenate([zeros, self.vali_lens], axis=0)
            if self.set_type == 2:
                self.target_data = np.array(self.target_data)
                self.naive_data = np.array(self.naive_data)
                self.target_data = self.target_data[self.instance_group]
                self.naive_data = self.naive_data[self.instance_group]

    def __getitem__(self, index):
        if self.set_type == 0:
            ind = np.abs(self.train_lens - index).argmin()
            if index == 0:
                ind = 0
            elif self.train_lens[ind] > index:
                ind = ind - 1
            data_x = self.data_x[ind]
            r_begin = index - int(self.train_lens[ind])
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = data_x[r_begin:r_end]
            return seq_x
        elif self.set_type == 1:
            ind = np.abs(self.vali_lens - index).argmin()
            if index == 0:
                ind = 0
            elif self.vali_lens[ind] > index:
                ind = ind - 1
            data_x = self.data_x[ind]
            r_begin = index - int(self.vali_lens[ind])
            r_end = r_begin + self.input_len + self.pred_len
            seq_x = data_x[r_begin:r_end]
            return seq_x
        else:
            data_x = self.data_x[index]
            target_data = self.target_data[index]
            seq_x = np.concatenate([data_x, target_data], axis=-1)
            if self.predict:
                naive_data = self.naive_data[index]
                seq_y = naive_data
                return seq_x, seq_y
            return seq_x

    def __len__(self):
        if self.set_type == 0:
            return int(self.train_lens[-1])
        elif self.set_type == 1:
            return int(self.vali_lens[-1])
        else:
            return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train_sample(self, sample_length):
        sample_data = np.array(self.sample_data)
        return sample_data

    def get_min_lens(self):
        return self.min_lens
