import torch.nn as nn
from pretrain_model.subLayer import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_ffn(enc_output)

        return enc_output


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_head, d_k, d_v, d_model, d_inner, dropout):
        super().__init__()
        self.enc_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

    def forward(self, en_input, mask):
        # -- Forward
        enc_output = self.enc_layer(en_input, mask=mask)

        return enc_output


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer
    """
    def __init__(self, d_k, d_v, d_model, d_inner, n_head, dropout):
        super().__init__()
        self.encoder = Encoder(d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)

    def forward(self, en_input, mask=None):
        if mask is not None:
            out = self.encoder(en_input, mask)
            return out


