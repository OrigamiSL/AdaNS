import torch
import torch.nn as nn
import torch.nn.functional as F
from AdaNS.embed import DataEmbedding
from AdaNS.encoder import EncoderLayer_Process, EncoderLayer_End
from utils.RevIN import RevIN


class AdaNS(nn.Module):
    def __init__(self, input_len, piece_len, pred_len, attn_pieces=4,
                 attn_nums1=3, attn_nums2=3, n_heads=4, d_model=16, Sample_strategy='Avg', dropout=0.0):
        super(AdaNS, self).__init__()
        self.input_len = input_len
        self.piece_len = piece_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.revin = RevIN(eps=1e-5)
        self.embed = DataEmbedding(2, d_model, dropout)

        encoder1 = [EncoderLayer_Process(d_model * 2 ** i, n_heads, attn_pieces,
                                         Sample_strategy, dropout, mode='Patch') for i in range(attn_nums1)]
        self.encoder1 = nn.ModuleList(encoder1)

        encoder2 = [EncoderLayer_Process(d_model * 2 ** (i + attn_nums1), n_heads, attn_pieces,
                                         Sample_strategy, dropout, mode='Intertwined') for i in range(attn_nums2 - 1)] + \
                   [EncoderLayer_End(d_model * 2 ** (attn_nums1 + attn_nums2 - 1), n_heads, attn_pieces, dropout)]
        self.encoder2 = nn.ModuleList(encoder2)
        self.F = nn.Flatten(start_dim=2)

        self.projection = nn.Linear((attn_nums2 + attn_nums1) * input_len * d_model // 2, pred_len)

    def forward(self, x_in):
        x_in = self.revin(x_in, 'norm')
        B, L, V = x_in.shape
        x_in_pieces = [x_in[:, :L//2, :].unsqueeze(-1), x_in[:, L//2:, :].unsqueeze(-1)]
        x_in_patch = torch.concat(x_in_pieces, dim=-1)
        x_bed = self.embed(x_in_patch)

        x_bed_list = []
        if len(self.encoder1) > 0:
            for enc1 in self.encoder1:
                x_bed = enc1(x_bed).clone()
                x_bed_list.append(self.F(x_bed))

        if len(self.encoder2) > 0:
            for enc2 in self.encoder2:
                x_bed = enc2(x_bed).clone()
                x_bed_list.append(self.F(x_bed))

        x_repr = torch.cat(x_bed_list, dim=-1)
        x_out = self.projection(x_repr).transpose(1, 2)
        x_out = self.revin(x_out, 'denorm')
        return x_out
