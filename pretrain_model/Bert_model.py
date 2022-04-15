import torch
import numpy as np
import torch.nn as nn
from pretrain_model import Constants
from pretrain_model.bert_embedding import BERTEmbedding
from pretrain_model.model import BERT, CrossAttention
from pretrain_model.task_model import Mask_skill, Pre_Difficulty, Mask_question


def get_non_pad_mask(seq):
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_pad_mask(seq_k, seq_q):
    """For masking out the padding part of key sequence."""

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x ls x lk
    return padding_mask


class BERTModel(nn.Module):
    """
      pretrain
    """

    def __init__(self, question_size, skill_size, d_model, n_layers, d_k, d_v, n_head, d_inner, dropout):
        super().__init__()
        self.q = question_size
        self.s = skill_size
        self.cross_attn = CrossAttention(temperature=np.power(d_model, 0.5), attn_dropout=dropout)
        self.bert_question = BERT(d_model=d_model, n_layers=n_layers, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.bert_skill = BERT(d_model=d_model, n_layers=n_layers, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.bert_embedding = BERTEmbedding(q_vocab_size=question_size+2, s_vocab_size=skill_size+2, embed_size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.task_predict_diff = Pre_Difficulty(d_model)
        self.task_mask_question = Mask_question(d_model, question_size+1)
        self.task_mask_skill = Mask_skill(d_model, skill_size+1)

    def forward(self, mask_question, mask_skill, question, total_skill):
        pad_mask_q = get_pad_mask(mask_question, mask_question)
        pad_mask_s = get_pad_mask(mask_skill, mask_skill)
        pad_difficult_question = get_pad_mask(question, question)
        pad_difficult_s = get_pad_mask(total_skill, total_skill)
        pad_difficult_qs = get_pad_mask(total_skill, question)
        print(self.q, self.s)
        print(mask_question[:3])
        print(mask_skill[:3])
        exit()
        mask_q_emb, mask_s_emb = self.bert_embedding(mask_question, mask_skill)
        exit()
        out_mask_q_emb = self.bert_question(mask_q_emb, mask=pad_mask_q)
        out_mask_s_emb = self.bert_skill(mask_s_emb, mask=pad_mask_s)

        q_emb, s_emb = self.bert_embedding(question, total_skill)

        out_q_emb = self.bert_question(q_emb, mask=pad_difficult_question)
        out_s_emb = self.bert_skill(s_emb, mask=pad_difficult_s)

        out_final_emb, cross_self_attn_list = self.cross_attn(out_q_emb, out_s_emb, out_s_emb, mask=pad_difficult_qs)

        seq_logit_mask_ques = self.task_mask_question(out_mask_q_emb)
        seq_logit_mask_skill = self.task_mask_skill(out_mask_s_emb)
        seq_logit_difficult_question = self.task_predict_diff(out_final_emb)

        return seq_logit_mask_ques, seq_logit_mask_skill, seq_logit_difficult_question

    def get_pretrain_question_embedding(self, question, skill):
        no_pad_mask = get_non_pad_mask(question)
        pad_mask_q = get_pad_mask(question, question)
        pad_mask_s = get_pad_mask(skill, skill)
        pad_mask_q_s = get_pad_mask(skill, question)

        question_emb, skill_emb = self.bert_embedding(question, skill)

        question_context_emb = self.bert_question(question_emb, mask=pad_mask_q)
        skill_context_emb = self.bert_skill(skill_emb, mask=pad_mask_s)
        final_question_emb, cross_self_attn_list = self.cross_attn(question_context_emb, skill_context_emb, skill_context_emb, mask=pad_mask_q_s)
        final_question_emb *= no_pad_mask

        return final_question_emb, cross_self_attn_list







