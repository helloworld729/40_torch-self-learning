#!/usr/bin/env python
"""Main training workflow"""
from __future__ import division

import argparse
import glob
import random

import torch
from pytorch_pretrained_bert import BertConfig

from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger

import os
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder','ff_actv', 'use_interval', 'rnn_size']


def baseline(args, cal_lead=False, cal_oracle=False):

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                  device, shuffle=False, is_test=True)

    trainer = build_trainer(args, None, None)
    #
    if (cal_lead):
        trainer.test(test_iter, 0, cal_lead=True)
    elif (cal_oracle):
        trainer.test(test_iter, 0, cal_oracle=True)

def train(args):
    init_logger(args.log_file)

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device == "cuda":
        # 表示有gpu
        torch.cuda.manual_seed(args.seed)

    # def model
    model = Summarizer(args, device, load_pretrained_bert=True)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
        model.load_cp(checkpoint)
        optim = model_builder.build_optim(args, model, checkpoint)
    else:
        optim = model_builder.build_optim(args, model, None)

    logger.info(model)
    trainer = build_trainer(args, model, optim)
    trainer.train(args.train_steps)

def validate(args, pt, step):
    logger.info('Loading checkpoint from %s' % pt)
    checkpoint = torch.load(pt, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config = config)
    model.load_cp(checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                   device, shuffle=False, is_test=True)
    trainer = build_trainer(args, model, None)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()

def test(args, pt, step):
    logger.info('Loading checkpoint from %s' % pt)
    checkpoint = torch.load(pt, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if k in model_flags:
            # 参数设置
            setattr(args, k, opt[k])
    print(args)

    config = BertConfig.from_json_file(args.bert_config_path)
    model = Summarizer(args, device, load_pretrained_bert=False, bert_config=config)
    model.load_cp(checkpoint)
    model.eval()

    # is_test为True的时候，会返回txt原文
    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                   device, shuffle=False, is_test=True)
    trainer = build_trainer(args, model, None)
    trainer.test(test_iter, step)

def testAll(args):
    timestep = 0
    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    xent_lst = []
    for i, cp in enumerate(cp_files):
        step = int(cp.split('.')[-2].split('_')[-1])
        xent = validate(args, cp, step)
        xent_lst.append((xent, cp))
        max_step = xent_lst.index(min(xent_lst))
        # if (i - max_step > 10):
        #     break
    xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
    logger.info('PPL %s' % str(xent_lst))
    for xent, cp in xent_lst:
        step = int(cp.split('.')[-2].split('_')[-1])
        test(args, cp, step)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='transformer', type=str, choices=[
                                  'classifier','transformer','rnn','baseline'])
    parser.add_argument("-mode", default='validate', type=str, choices=[
                                'train', 'validate', 'test', 'lead', 'oracle'])
    parser.add_argument("-bert_data_path", default='../bert_data/cnndm')
    parser.add_argument("-model_path", default='../models/', help="存储checkpoint")
    parser.add_argument("-result_path", default='../results/cnndm', help="存储两个摘要")
    parser.add_argument("-temp_dir", default='../temp',
                        help="存储bert预处理模型和rouge的临时数据")
    parser.add_argument("-bert_config_path", default='../bert_config_uncased_base.jsonData')

    parser.add_argument("-batch_size", default=5000, type=int,
                        help="不是样本的个数，而是token的个数，根据显存设置# 18500")

    parser.add_argument("-use_interval", default=True, help="句见seg标记")
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=4, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)

    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", default=True)
    parser.add_argument("-dropout", default=0.1, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=2e-4, type=float)
    parser.add_argument("-beta1", default=0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=10000, type=int)
    parser.add_argument("-max_grad_norm", default=0.1, type=float,
                        help="当默认!=0时，可以启动梯度修剪")

    parser.add_argument("-save_checkpoint_steps", default=2500, type=int)  # 5
    parser.add_argument("-report_every", default=1, type=int)
    parser.add_argument("-train_steps", default=100000, type=int)  # 1000
    parser.add_argument("-recall_eval", default=False)

    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=527, type=int)

    parser.add_argument("-test_all", default=True)
    parser.add_argument("-test_from", default="../models/" + "model_step_")
    parser.add_argument("-train_from", default='',
                        help="可以选择从某个checkpoint继续训练")
    parser.add_argument("-report_rouge", default=True)
    parser.add_argument("-block_trigram", default=True, help="摘要去冗相关")

    args = parser.parse_args()

    init_logger(args.log_file)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'validate':
        testAll(args)
    elif args.mode == 'lead':
        baseline(args, cal_lead=True)
    elif args.mode == 'oracle':
        baseline(args, cal_oracle=True)
    elif args.mode == 'test':
        step = "2500"
        cp = args.test_from + step + ".pt"
        test(args, cp, int(step))

## end ##
