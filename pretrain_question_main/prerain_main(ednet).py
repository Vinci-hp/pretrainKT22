import argparse
import time
import os
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.utils.data
from pretrain_model.Bert_model import BERTModel
from pretrain_question_main.pretrain_dataset import getData, PretrainDataSet
from pretrain_question_main.pretraining import pretrain
from pretrain_question_main.optim_schedule import ScheduledOptim
from tensorboardX import SummaryWriter
from util import setup_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
writer = SummaryWriter('runs/auc_ednet')
setup_seed(42)


def get_embedding(model, question, skill):
    """ get_embedding """
    model.eval()
    with torch.no_grad():
        # forward
        q_embed_data = model.get_embedding(question, skill)
        return q_embed_data


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-batch_size', type=int, default=128)

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-save_model', default='../savaMadel/')

    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-q_sum', type=int, default=12372)
    parser.add_argument('-s_sum', type=int, default=1901)
    parser.add_argument('-s_size', type=int, default=188)
    parser.add_argument('-log', default=None)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-sq_len', type=int, default=20)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # ========= Loading Dataset =========
    print('*' * 30, '数据加载中......', '*' * 30)
    start = time.time()
    # =============data====================
    train_data, skill_data, diff_skill, dif = getData('ednet', opt.sq_len)
    pretrain_data = torch.utils.data.DataLoader(PretrainDataSet(train_data, skill_data, diff_skill, dif, opt.q_sum, opt.s_size, opt.sq_len), batch_size=opt.batch_size, shuffle=True)

    print('*' * 30, '耗时:', '{time:3.3f}s'.format(time=(time.time()-start)), '*' * 30)
    print('*' * 30, '参数打印中......', '*' * 30)
    print(opt)
    print('*' * 30, '训练模型加载中......', '*' * 30)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    # =============pretain===============================
    print('pretrain............')
    pretrain_bert = BERTModel(q_size=opt.q_sum, s_size=opt.s_size, s_label_size=opt.s_size, d_k=opt.d_k, d_v=opt.d_v, d_model=opt.d_model, d_inner=opt.d_inner_hid, n_layers=opt.n_layers, n_head=opt.n_head, dropout=opt.dropout).to(device)
    optimizer = ScheduledOptim(optim.Adam(pretrain_bert.parameters(), betas=(0.9, 0.98), eps=1e-09), opt.d_model, opt.n_warmup_steps)
    pretrain(pretrain_bert, pretrain_data, optimizer, device, opt, 'ednet', writer)


if __name__ == '__main__':
    main()
