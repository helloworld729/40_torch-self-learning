# -*- coding:utf-8 -*-
import argparse
import time

from others.logging import init_logger
from prepro import data_builder


def do_format_to_lines(args, log):
    t1 = time.time()
    data_builder.format_to_lines(args, log)
    t2 = time.time()
    print("format_to_lines 用时：", t2-t1)
    log.info("format_to_lines 用时：{}".format(t2 - t1))

def do_tokenize(args, log):
    t1 = time.time()
    data_builder.tokenize(args, log)
    t2 = time.time()
    print("tokenize 用时：", t2-t1)
    log.info("tokenize 用时：       {}".format(t2 - t1))

def do_format_to_bert(args, log):
    t1 = time.time()
    data_builder.format_to_bert(args, log)
    t2 = time.time()
    print("format_to_bert 用时：", t2-t1)
    log.info("format_to_bert 用时： {}".format(t2 - t1))




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str, help='format_to_lines or format_to_bert')
    parser.add_argument("-oracle_mode", default='greedy', type=str, help='how to generate oracle summaries, greedy or combination, combination will generate more accurate oracles but take much longer time.')
    parser.add_argument("-dataMap", default='../dataMap/')
    parser.add_argument("-dataRawCnn", default='../dataRaw/cnn_stories_tokenized/')      # 原始数据
    parser.add_argument("-dataRawDm", default='../dataRaw/dm_stories_tokenized/')      # 原始数据
    parser.add_argument("-dataBert", default='../dataBert/')    # 最终地址
    parser.add_argument("-dataJson", default='../dataJson/')    # json地址
    parser.add_argument("-dataToken", default='../dataToken/')  # token地址

    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=512, type=int)  # 200

    parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)

    parser.add_argument('-log_file', default='../logs/datapre.log')

    parser.add_argument('-dataset', default='', help='train, valid or test, defaul will process all datasets')

    parser.add_argument('-n_cpus', default=2, type=int)

    args = parser.parse_args()
    log = init_logger(args.log_file)
    # eval('data_builder.'+args.mode + '(args)')
    # 任务开始
    args.mode = "tokenize"
    log.info("开始训练")
    do_tokenize(args, log)
    args.mode = "format_to_lines"
    do_format_to_lines(args, log)
    args.mode = "format_to_bert"
    do_format_to_bert(args, log)


