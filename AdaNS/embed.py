import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[:, :x.size(1)].unsqueeze(-1)
        return pos.permute(0, 3, 1, 2)


class TokenEmbedding(nn.Module):
    def __init__(self, in_piece, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv2d(in_channels=in_piece, out_channels=d_model,
                                   kernel_size=(1, 1), padding=0)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 3, 1, 2))  # B, C, L ,V
        return x.permute(0, 3, 2, 1)  # B, V, L, C


class DataEmbedding(nn.Module):
    def __init__(self, in_piece, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(in_piece=in_piece, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B, L, V, C = x.shape
        x = self.value_embedding(x) + self.position_embedding(x).expand(B, V, L, self.d_model)
        return self.dropout(x)
