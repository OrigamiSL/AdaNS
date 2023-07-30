import torch
import torch.nn as nn
import torch.nn.functional as F
from AdaNS.attn import AttentionLayer


def intertwined_slice(x):
    x1 = x[:, :, 0::2, :]  # B, V, L, C
    x2 = x[:, :, 1::2, :]
    x_intertwined = torch.cat([x1, x2], dim=-1)  # B, V, L/2, 2C
    return x_intertwined


def Patch_slice(x):
    x1 = x[:, :, :x.shape[-2] // 2, :]  # B, V, L, C
    x2 = x[:, :, x.shape[-2] // 2:, :]
    x_patch = torch.cat([x1, x2], dim=-1)  # B, V, L/2, 2C
    return x_patch


class Sample_layer(nn.Module):
    def __init__(self, Sample_strategy, d_model):
        super( Sample_layer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)) \
            if Sample_strategy == 'Max' else \
            nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        self.conv = nn.Conv2d(in_channels=d_model, out_channels=2 * d_model, kernel_size=(1, 1))

    def forward(self, x):
        x = self.pool(x.transpose(1, -1))
        x = self.conv(x).transpose(1, -1)
        return x


class EncoderLayer_Process(nn.Module):
    def __init__(self, d_model, n_heads, attn_pieces=4, Sample_strategy='Max', dropout=0.1, mode='Patch'):
        super(EncoderLayer_Process, self).__init__()
        d_ff = 4 * d_model // 2
        self.d_model = d_model
        self.attn_pieces = attn_pieces
        self.mode = mode
        self.attention_local = AttentionLayer(d_model // 2, n_heads, attn_pieces, dropout)
        self.attention_neighbour = AttentionLayer(d_model // 2, n_heads, attn_pieces, dropout)
        self.conv1 = nn.Conv2d(in_channels=d_model // 2, out_channels=d_ff, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model // 2, kernel_size=(1, 1))
        self.projection = nn.Linear(d_model, d_model)
        self.sample_layer = Sample_layer(Sample_strategy, d_model // 2)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.norm2 = nn.LayerNorm(d_model // 2)
        self.norm3 = nn.LayerNorm(d_model // 2)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, ):
        x_attn = x[:, :, :, :self.d_model // 2]
        x_sample = x[:, :, :, self.d_model // 2:]

        x_1 = self.attention_local(
            x_attn, x_attn, x_attn, flag='local'
        )
        x_attn = self.norm1(x_attn + self.dropout(x_1))  # B, V, L ,C

        x_2 = self.attention_neighbour(
            x_attn, x_attn, x_attn, flag='neighbour'
        )
        x_attn = x_attn + self.dropout(x_2)  # B, V, L ,C

        y_attn = x_attn = self.norm2(x_attn)

        y_attn = self.dropout(self.activation(self.conv1(y_attn.transpose(-1, 1))))
        y_attn = self.dropout(self.conv2(y_attn).transpose(-1, 1))
        x_norm = self.norm3(x_attn + y_attn)

        if self.mode == 'Patch':
            x_deep = Patch_slice(x_norm)
        else:
            x_deep = intertwined_slice(x_norm)
        x_deep = self.norm4(x_deep + self.projection(x_deep))

        x_sample = self.sample_layer(x_sample)

        x_total = torch.cat([x_deep, x_sample], dim=-1)
        return x_total


class EncoderLayer_End(nn.Module):
    def __init__(self, d_model, n_heads, attn_pieces=6, dropout=0.1):
        super(EncoderLayer_End, self).__init__()
        d_ff = 4 * d_model
        self.attn_pieces = attn_pieces
        self.attention = AttentionLayer(d_model, n_heads, attn_pieces, dropout)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=(1, 1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x, ):
        new_x = self.attention(
            x, x, x, flag='full'
        )
        x = x + self.dropout(new_x)  # B, V, L ,C

        y = x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x_norm = self.norm2(x + y)

        return x_norm
