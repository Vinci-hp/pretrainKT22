''' Define the bert model '''
import torch.nn as nn
import torch
from pretrain_model.transformer import TransformerBlock
import numpy as np


class CrossAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2) #d

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, d_model, n_layers, d_k, d_v, n_head, d_inner, dropout):
        super().__init__()
        self.transformers = nn.ModuleList([TransformerBlock(d_k, d_v, d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, input_data, mask=None):
        if mask is not None:
            output = input_data
            for transformer in self.transformers:
                output = transformer(output, mask)
            return output