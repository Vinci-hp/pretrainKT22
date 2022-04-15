import torch.nn as nn
import torch
from pretrain_model import Constants


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        TokenEmbedding : normal embedding matrix
    """

    def __init__(self, q_vocab_size, s_vocab_size, embed_size, dropout):
        super().__init__()
        self.tokenQ = nn.Embedding(q_vocab_size, embed_size, padding_idx=Constants.PAD)
        self.tokenS = nn.Embedding(s_vocab_size, embed_size, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_1, input_2):
        input1_emb = self.tokenQ(input_1)
        input2_emb = self.tokenS(input_2)
        return self.dropout(input1_emb), self.dropout(input2_emb)





