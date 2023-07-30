import torch
import torch.nn as nn
import math


class FullAttention(nn.Module):
    def __init__(self, attn_pieces=6, dropout=0.1):
        super(FullAttention, self).__init__()
        self.attn_pieces = attn_pieces
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, flag):
        B, V, L, H, E = queries.shape
        scale = 1. / math.sqrt(E)
        if flag == 'full':
            scores = torch.einsum("bvlhe,bvshe->bvhls", queries, keys)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            Out = torch.einsum("bvhls,bvshd->bvlhd", A, values)
        elif flag == 'local':
            queries = queries.permute(0, 1, 3, 2, 4).contiguous(). \
                view(B, V, H, L // self.attn_pieces, self.attn_pieces, E)
            keys = keys.permute(0, 1, 3, 2, 4).contiguous(). \
                view(B, V, H, L // self.attn_pieces, self.attn_pieces, E)
            values = values.permute(0, 1, 3, 2, 4).contiguous(). \
                view(B, V, H, L // self.attn_pieces, self.attn_pieces, E)
            scores = torch.einsum("bvhple,bvhpse->bvhpls", queries, keys)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            Out = torch.einsum("bvhpls,bvhpse->bvhple", A, values)
            Out = Out.permute(0, 1, 2, 5, 3, 4).contiguous().view(B, V, H, E, L).permute(0, 1, 4, 2, 3)
        elif flag == 'neighbour':
            se_queries = torch.cat([queries[:, :, :self.attn_pieces // 2], queries[:, :, -self.attn_pieces // 2:]],
                                   dim=2).clone()
            se_keys = torch.cat([keys[:, :, :self.attn_pieces // 2], keys[:, :, -self.attn_pieces // 2:]],
                                dim=2).clone()
            se_values = torch.cat([values[:, :, :self.attn_pieces // 2], values[:, :, -self.attn_pieces // 2:]],
                                  dim=2).clone()
            se_scores = torch.einsum("bvlhe,bvshe->bvhls", se_queries, se_keys)
            se_A = self.dropout(torch.softmax(scale * se_scores, dim=-1))
            se_Out = torch.einsum("bvhls,bvshd->bvlhd", se_A, se_values)

            queries = queries[:, :, self.attn_pieces // 2: L - self.attn_pieces//2].permute(0, 1, 3, 2, 4).\
                contiguous().view(B, V, H, L // self.attn_pieces - 1, self.attn_pieces, E)
            keys = keys[:, :, self.attn_pieces // 2: L - self.attn_pieces//2].permute(0, 1, 3, 2, 4).\
                contiguous().view(B, V, H, L // self.attn_pieces - 1, self.attn_pieces, E)
            values = values[:, :, self.attn_pieces // 2: L - self.attn_pieces//2].permute(0, 1, 3, 2, 4).\
                contiguous().view(B, V, H, L // self.attn_pieces - 1, self.attn_pieces, E)
            scores = torch.einsum("bvhple,bvhpse->bvhpls", queries, keys)
            A = self.dropout(torch.softmax(scale * scores, dim=-1))
            Out = torch.einsum("bvhpls,bvhpse->bvhple", A, values)
            Out = Out.permute(0, 1, 2, 5, 3, 4).contiguous().view(B, V, H, E, L - self.attn_pieces).permute(0, 1, 4, 2, 3)
            Out = torch.cat([se_Out[:, :, :self.attn_pieces // 2], Out, se_Out[:, :, -self.attn_pieces // 2:]], dim=2)
        else:
            print('no flag!')
            exit(-1)

        return Out.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, attn_pieces, dropout):
        super(AttentionLayer, self).__init__()
        self.inner_attention = FullAttention(attn_pieces, dropout)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, flag='full'):
        B, V, L, _ = queries.shape
        _, V, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, V, L, H, -1)
        keys = self.key_projection(keys).view(B, V, S, H, -1)
        values = self.value_projection(values).view(B, V, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
            flag=flag
        )
        out = out.view(B, V, L, -1)

        return self.out_projection(out)
