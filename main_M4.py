import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import time, math

from exp.exp_model_M4 import Exp_Model
from utils.tools import *
from data.data_loader import Dataset_m4

parser = argparse.ArgumentParser(description='[AdaNS]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, '
                         'S: univariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or M task')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--tuned_checkpoints', type=str, default='./tuned_checkpoints/',
                    help='location of tuned model checkpoints')

parser.add_argument('--sample_train', type=int, default=192,
                    help='the length of sampled sequence to seek fs and Ts')
parser.add_argument('--lmb', type=int, default=129600, help='year=6.25, season=1600, month=129600')
parser.add_argument('--c', type=int, default=2)

parser.add_argument('--input_len', type=int, default=96, help='input sequence length')
parser.add_argument('--piece_len', type=int, default=24, help='the length of Ts, acquired by later codes')
parser.add_argument('--Period_Times', type=int, default=8, help='the value of I')
parser.add_argument('--Sample_strategy', type=str, default='Avt',
                    help='sampling strategy in SCMA')
parser.add_argument('--short_input_len', type=int, default=24, help='short input length for variable with fs=0')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence')

parser.add_argument('--c_in', type=int, default=7, help='variable number')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--w_random', action='store_true', help='whether to train with partial variables', default=False)
parser.add_argument('--rand_thres', type=int, default=100,
                    help='threshold variable number determining whether to use w_random')
parser.add_argument('--factor', type=int, default=5, help='b')

parser.add_argument('--d_model', type=int, default=32, help='hidden dimension of model')
parser.add_argument('--attn_pieces', type=int, default=6, help='number of patches')
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--attn_nums1', type=int, default=3, help='attn blocks of the first stage, acquired adaptively')
parser.add_argument('--attn_nums2', type=int, default=3, help='attn blocks of the first stage, acquired adaptively')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=1, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--save_loss', action='store_true', help='whether saving results and checkpoints', default=False)
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--train', action='store_true',
                    help='whether to train'
                    , default=False)
parser.add_argument('--reproducible', action='store_true',
                    help='whether to make results reproducible'
                    , default=False)
parser.add_argument('--load_tuned', action='store_true',
                    help='whether to load the tuned checkpoints'
                    , default=False)

parser.add_argument('--freq', type=str, default='Daily', help='sampling frequency')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() else False

if args.reproducible:
    np.random.seed(4321)  # reproducible
    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    torch.backends.cudnn.deterministic = True
else:
    torch.backends.cudnn.deterministic = False

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if args.use_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'M4_Yearly': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
    'M4_Quarterly': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
    'M4_Monthly': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
    'M4_Weekly': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
    'M4_Daily': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
    'M4_Hourly': {'root_path': './data/M4/', 'M': [1, 1], 'S': [1, 1]},
}

if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.root_path = data_info['root_path']
    args.c_in, args.c_out = data_info[args.features]

args.target = args.target.replace('/r', '').replace('/t', '').replace('/n', '')
if 'M4' in args.data:
    _, args.freq = args.data.split('_')
type_map = {'Yearly': 1, 'Quarterly': 4, 'Monthly': 12, 'Weekly': 1, 'Daily': 1, 'Hourly': 24}
args.frequency = type_map[args.freq]

lr = args.learning_rate

data_dict = {
    'M4_Yearly': Dataset_m4,
    'M4_Quarterly': Dataset_m4,
    'M4_Monthly': Dataset_m4,
    'M4_Weekly': Dataset_m4,
    'M4_Daily': Dataset_m4,
    'M4_Hourly': Dataset_m4,
}
Data = data_dict[args.data]

path = './result_m4.log'
instance_groups = []
num_index = None
total_index = []

data_set = Data(
    root_path=args.root_path,
    flag='train',
    size=[args.input_len, args.pred_len],
    freq=args.freq,
    sample_length=args.sample_train,
    sample=True
)
x = np.linspace(0, args.sample_train-1, args.sample_train)
sample_train_data = data_set.get_train_sample(args.sample_train)
preprocess_sample = preprocess_m4(sample_train_data, args.lmb, args.c)
freq_domain_sample = np.fft.fft(preprocess_sample, axis=1)
# plt.plot(x, preprocess_sample[0, :])
# plt.plot(x, np.abs(freq_domain_sample[0, :]))
# plt.show()



min_lens = data_set.get_min_lens()
partial_freq_domain_sample = np.abs(freq_domain_sample)[:, :args.sample_train // 2 + 1]
max_index = np.argmax(partial_freq_domain_sample, axis=1)
input_lens = np.where(args.sample_train // max_index == args.sample_train / max_index, args.Period_Times * (args.sample_train // max_index), 2 * args.sample_train)
index_less = np.where(input_lens >= min_lens, -1, max_index)
num_index = sorted(list(set(max_index)))

args.instance_num = len(max_index)
zero_f_num = 0
i_index = 0
for each_index in num_index:
    each_instance_group = np.array(np.where(max_index == each_index)[0])
    if each_index == 0:
        zero_f_num = each_instance_group.shape[0]
        for i in range(each_instance_group.shape[0]):
            zero_array = np.array([0])
            zero_array[0] = each_instance_group[i]
            instance_groups.append(zero_array)
            total_index.append(each_index)
            print("group_{}: f_0, num_1".format(i_index))
            with open(path, "a") as f:
                f.write("group_{}: f_0, num_1".format(i_index) + '\n')
                f.flush()
                f.close()
            i_index += 1

    else:
        instance_groups.append(each_instance_group)
        total_index.append(each_index)
        print("group_{}: f_{}, num_{}".format(i_index, each_index, each_instance_group.shape[0]))
        with open(path, "a") as f:
            f.write("group_{}: f_{}, num_{}".format(i_index, each_index, each_instance_group.shape[0]) + '\n')
            f.flush()
            f.close()
        i_index += 1

mse_list = []
mae_list = []
smape_list = []
owa_list = []
if args.load_tuned:
    args.itr = 1

for ii in range(args.itr):
    group_index = 0
    tg_smape = 0
    tg_owa = 0
    accum_ins_num = 0
    original_instance = args.instance_num
    for ins_group in instance_groups:
        min_lens_group = min(min_lens[var] for var in ins_group)
        args.instance_num = ins_group.shape[0]
        cur_freq = 1e6
        group_less = index_less[ins_group]
        ins_group_train = ins_group
        if -1 in group_less:
            if ins_group.shape[0] > 1:
                ins_group_train = np.delete(ins_group, np.where(group_less < 0))
            else:
                total_index[group_index] = 0
        if accum_ins_num + ins_group.shape[0] > zero_f_num and total_index[group_index]:
            cur_freq = total_index[group_index]
            if args.sample_train // cur_freq == args.sample_train / cur_freq:
                args.input_len = args.Period_Times * (args.sample_train // cur_freq)
                args.piece_len = args.sample_train // cur_freq
                args.d_model = max(args.n_heads * 2, args.Period_Times * 2)
                args.attn_nums1 = int(math.log2(args.Period_Times)) - 1
                args.attn_nums2 = max(int(math.log2(args.sample_train // cur_freq // args.attn_pieces + 1e-6)), 1)
            else:
                args.input_len = 2 * args.sample_train
                args.piece_len = args.sample_train
                args.d_model = args.n_heads * 2
                args.attn_nums1 = 0
                args.attn_nums2 = int(math.log2(args.input_len // args.attn_pieces))

            setting = '{}_ft{}_ll{}_pl{}_blf{}_bls{}_group_{}_{}'. \
                format(args.data, args.features, args.input_len, args.pred_len,
                       args.attn_nums1, args.attn_nums2, group_index, ii)
        else:
            args.input_len = 2 * args.short_input_len
            args.piece_len = args.short_input_len
            args.d_model = args.n_heads * 2
            args.attn_nums1 = 0
            args.attn_nums2 = int(math.log2(args.input_len // args.attn_pieces))
            setting = '{}_ft{}_ll{}_pl{}_blf{}_bls{}_group_{}_{}'. \
                format(args.data, args.features, args.input_len, args.pred_len,
                       args.attn_nums1, args.attn_nums2, group_index, ii)
        print(group_index, args.input_len, args.sample_train, cur_freq)

        print('Args in experiment:')
        print(args)
        Exp = Exp_Model
        exp = Exp(args)  # set experiments
        if args.train and not args.load_tuned:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            try:
                if group_index == 253:
                    exp.train(setting, ins_group_train)
            except KeyboardInterrupt:
                print('-' * 99)
                print('Exiting from training early')

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if group_index == 253:
            g_smape, g_owa = exp.test(setting, ins_group, group_index=group_index,
                                    load=True, save_loss=args.save_loss, load_tuned=args.load_tuned)
            tg_smape += g_smape * args.instance_num / original_instance
            tg_owa += g_owa * args.instance_num / original_instance

        torch.cuda.empty_cache()
        args.instance_num = original_instance
        args.learning_rate = lr
        group_index += 1
        accum_ins_num += ins_group.shape[0]

    with open(path, "a") as f:
        f.write(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        f.write('|{}_{}|pred_len{}|smape:{}, owa:{}'.
                format(args.data, args.features, args.pred_len, tg_smape, tg_owa) + '\n')
        f.flush()
        f.close()
    smape_list.append(tg_smape)
    owa_list.append(tg_owa)

smape = np.asarray(smape_list)
owa = np.asarray(owa_list)
avg_smape = np.mean(smape)
std_smape = np.std(smape)
avg_owa = np.mean(owa)
std_owa = np.std(owa)
print('|Mean|smape:{}, owa:{}|Std|smape:{}, owa:{}'.
      format(avg_smape, avg_owa, std_smape, std_owa))

with open(path, "a") as f:
    f.write('|{}_{}|pred_len{}: '.
            format(args.data, args.features, args.pred_len) + '\n')
    f.write('|Mean|smape:{}, owa:{}|Std|smape:{}, owa:{}'.
            format(avg_smape, avg_owa, std_smape, std_owa) + '\n')
    f.flush()
    f.close()
