'''This script handling the training process.'''
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from dataset import TranslationDataset, paired_collate_fn
import math, time, torch, argparse, torch.utils.data, random, numpy as np

# ###################################### 损失计算 ##########################################################
def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct

def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss

# ###################################### epoch训练与验证 ###################################################
def prepare_dataloaders(data, source, opt):
    data_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data[source]['src'],
            tgt_insts=data[source]['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)
    return data_loader

def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase '''
    # 使得 requires_grad为真
    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]  # 从第一列开始，因为都是以2(BOS)作为开头

        # 前向函数
        optimizer.zero_grad()
        pred = model(src_seq, src_pos, tgt_seq, tgt_pos)

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        # optimizer.step_and_update_lr()
        optimizer.step()

        # note keeping
        total_loss += loss.item()

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()  # 不产生梯度

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]
            # print("\n验证集target")
            # print(gold)
            # forward
            pred = model(src_seq, src_pos, tgt_seq, tgt_pos)
            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, device, opt, lr, optim_index):  # 模型 数据 优化器 参数
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    optimiz = ["sgd", "adam"]
    if opt.log:
        log_train_file = opt.log + '.train.log' + "."+lr + "."+str(optimiz[optim_index])
    if opt.has_validation:
        log_valid_file = opt.log + '.valid.log' + "."+lr + "."+str(optimiz[optim_index])

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
    if opt.has_validation:
        with open(log_valid_file, 'w') as log_vf:
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []  # 验证集准确率
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()

        model.zero_grad()

        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)

        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        # 如果没有验证集，就用测试集作为结果
        valid_loss, valid_accu = eval_epoch(model, validation_data, device) if \
                                     opt.has_validation and validation_data else (train_loss, train_accu)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()  # 状态字典
        checkpoint = {                         #
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file:
            with open(log_train_file, 'a') as log_tf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
        if log_valid_file and opt.has_validation:
            with open(log_valid_file, 'a') as log_vf:
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default='data/save_file/file_saved.txt')  # 数据
    parser.add_argument('-has_validation', default=True)  # 数据
    parser.add_argument('-epoch', type=int, default=6)  # 10
    parser.add_argument('-batch_size', type=int, default=64)  # 64
    parser.add_argument('-d_model', type=int, default=256)  # 512
    parser.add_argument('-d_inner_hid', type=int, default=1024)  # 2048
    parser.add_argument('-d_k', type=int, default=64)  # 64
    parser.add_argument('-d_v', type=int, default=64)  # 64
    parser.add_argument('-n_head', type=int, default=8)  # 8
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=100)  # 4000
    parser.add_argument('-dropout', type=float, default=0.1)  # 0.1
    # parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-embs_share_weight', default=True)
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-log', default='log/transformer')
    parser.add_argument('-save_model', default='weights/transformer')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='all')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true',)
    parser.add_argument('-seed', default=37)
    opt = parser.parse_args()
    # action类型的参数，是指我们要在命令行中输入第一参数，例如：
    # parser.add_argument('-proj_share_weight', action='store_true')
    # 我们在命令行中输入 -proj_share_weight，那么proj_share_weight = True

    opt.cuda = True if torch.cuda.is_available() else False
    opt.d_word_vec = opt.d_model

    def set_seed(opt):
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        if opt.cuda > 0:
            torch.cuda.manual_seed_all(opt.seed)
    set_seed(opt)


    # ========= Loading Dataset ========= #
    data = torch.load(opt.data)
    # 句子长度上限,用于截断
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    # 返回两个dataloader，已经包含batch_size，collect_fn 等信息
    training_data   = prepare_dataloaders(data, "train", opt)
    validation_data = prepare_dataloaders(data, "valid", opt) if opt.has_validation else None
    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # ========= Preparing Model ========= #
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')


    # optimizer = ScheduledOptim(
    #     optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()), lr=1e-9,
    #         betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-6),  opt.d_model, opt.n_warmup_steps)
    for optim_index in [0, 1]:
        for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
            optimizer1 = optim.SGD(filter(lambda x: x.requires_grad, transformer.parameters()), lr=lr, momentum=0.9)
            optimizer2 = ScheduledOptim(
                optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()), lr=lr,
                           betas=(0.9, 0.98), eps=1e-09, weight_decay=1e-6), opt.d_model, opt.n_warmup_steps)
            optimizer = [optimizer1, optimizer2][optim_index]

            transformer = Transformer(
                opt.src_vocab_size,
                opt.tgt_vocab_size,
                opt.max_token_seq_len,
                tgt_emb_prj_weight_sharing=opt.proj_share_weight,
                emb_src_tgt_weight_sharing=opt.embs_share_weight,
                d_k=opt.d_k,
                d_v=opt.d_v,
                d_model=opt.d_model,
                d_word_vec=opt.d_word_vec,
                d_inner=opt.d_inner_hid,
                n_layers=opt.n_layers,
                n_head=opt.n_head,
                dropout=opt.dropout).to(device)


            # 模型，数据，优化器，设备，参数类
            train(transformer, training_data, validation_data, optimizer, device, opt, str(lr), optim_index)

if __name__ == '__main__':
    main()
