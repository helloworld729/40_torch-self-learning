# -*- coding:utf-8 -*-
import gc
import glob
import hashlib
import itertools
import json
import os
import re
import subprocess
import time
from os.path import join as pjoin

import torch
from multiprocess import Pool
# from multiprocess import pool as Pool
from pytorch_pretrained_bert import BertTokenizer

from others.logging import logger
from others.myUtils import clean
from prepro.utils import _get_word_ngrams

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if (flag):
            tgt.append(tokens)
            flag = False
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def cal_rouge(evaluated_ngrams, reference_ngrams):
    # 计算两个集合的 精确率、召回率、F1
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    max_idx = (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))

# 两个输入都是二维列表，元素为token构成的列表
def greedy_selection(doc_sent_list, abstract_sent_list, summary_size, args):
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])  # 摘要句融合为一维列表
    abstract = _rouge_clean(' '.join(abstract)).split()
    sents = [_rouge_clean(' '.join(s)).split() for s in doc_sent_list]  # 原文 二维列表
    # _get_word_ngrams 的第二个参数是 二维列表
    evaluated_1grams = [_get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = [_get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = _get_word_ngrams(2, [abstract])

    selected = []
    totalScores = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        stepScores = []
        for i in range(len(sents)):  # 遍历原文
            if (i in selected) or (len(sents[i]) <= args.min_src_ntokens):  # Todo
                stepScores.append(0.0)
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            stepScores.append(rouge_score)
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if cur_id == -1: return selected, totalScores

        assert cur_id <= len(sents) -1
        selected.append(cur_id)
        max_rouge = cur_max_rouge
        assert len(stepScores) == len(sents)
        # 打分策略：
        mins, maxs = min(stepScores), max(stepScores)
        stepScores = [(x-mins)/ (maxs-mins) for x in stepScores]
        totalScores.append(stepScores)
        assert len(totalScores) <= summary_size

        # return sorted(selected)  # 完全抽取的话 不用考虑顺序
    return selected, totalScores  # 就要生成需要考虑顺序RBX

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

class BertData():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def getNewOracle(self, oracle_ids, dels):
        counts = {i: 0 for i in range(len(oracle_ids))}
        for pos in dels:
            for i in range(len(oracle_ids)):
                if oracle_ids[i] > pos:
                    counts[i] += 1
        for i, count in counts.items():
            oracle_ids[i] -= count
        return oracle_ids


    def preprocess(self, src, tgt, oracle_ids):
        # 可能造成oracleIds与labels不一致的地方分析：
        # labels的作用：在训练的时候和句子对用，这一块是没有问题的
        # oracleIds的作用：1初始化labels，2、计算oracle得分(这是需要oracle对原文索引而不是处理后的)

        # src是二维列表，每一个元素是一个句子单词构成的列表
        print("\n\n")
        print("**********************************************************")
        print(oracle_ids)

        if len(src) == 0:
            return None

        # 全文列表
        original_src_txt = [' '.join(s) for s in src]
        # 句子数目
        labels = [0] * len(src)
        # 每个句子的标记
        for l in oracle_ids:
            assert l <= len(labels)-1
            labels[l] = 1
        print("labels0: ", labels)
        # 筛选掉很短的句子
        idxs = []
        dels = []
        for i, s in enumerate(src):
            if len(s) > self.args.min_src_ntokens:
                idxs.append(i)
            else:
                dels.append(i)
        newOracle = self.getNewOracle(oracle_ids, dels)
        # idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]  # 句子索引

        # src与labels 句子长度级别的截断 且对其
        src = [src[i][:self.args.max_src_ntokens] for i in idxs]  # 句子截断后重新构成二维列表
        labels = [labels[i] for i in idxs]
        print("dels: ", dels)
        print("labels1: ", labels)

        # src与labels 最后 句子数目级别的截断
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]
        print("句子数目截断后 labels1: ", labels)

        if len(src) < self.args.min_nsents:
            return None
        if len(labels) == 0:
            return None

        # 全文一维列表，元素为 用空格隔开的 句子str
        src_txt = [' '.join(sent) for sent in src]
        # text = [' '.join(ex['src_txt'][i].split()[:self.args.max_src_ntokens]) for i in idxs]
        # text = [_clean(t) for t in text]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

        # 根据句子长度，设置segment
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        # cls的位置
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]
        newOracle = [d for d in newOracle if d <= len(labels)-1]
        print("newOracle: ", newOracle)

        print("文章总长截断后labels:   ", labels)
        print("**********************************************************")
        print("\n\n")

        tgt_txt = '<q>'.join([' '.join(tt) for tt in tgt])
        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt, newOracle

def tokenize(args, log):
    stories_dir1 = os.path.abspath(args.dataRawCnn)
    stories_dir2 = os.path.abspath(args.dataRawDm)
    # tokenized_stories_dir = os.path.abspath(args.save_path)
    tokenized_stories_dir = os.path.abspath(args.dataToken)

    print("Preparing to tokenize %s to %s..." % (stories_dir1, tokenized_stories_dir))
    print("Preparing to tokenize %s to %s..." % (stories_dir2, tokenized_stories_dir))

    stories1 = os.listdir(stories_dir1)
    stories2 = os.listdir(stories_dir2)
    # make IO list file
    print("Making list of files to tokenize...")
    with open(args.dataMap + "mapping_for_train.txt", "w") as ftrain, \
         open(args.dataMap + "mapping_for_valid.txt", "w") as fvalid, \
         open(args.dataMap + "mapping_for_test.txt", "w")  as ftest:

        i, totalLen = 1, len(stories1)

        for s in stories1:
            if (not s.endswith('story')):
                continue
            if i <= 90266:
                ftrain.write("%s\n" % (os.path.join(stories_dir1, s)))
            elif i <= 90266+1220:
                fvalid.write("%s\n" % (os.path.join(stories_dir1, s)))
            else:
                ftest.write("%s\n" % (os.path.join(stories_dir1, s)))
            i += 1

        i, totalLen = 1, len(stories2)

        for s in stories2:
            if (not s.endswith('story')):
                continue
            if i <= 196961:
                ftrain.write("%s\n" % (os.path.join(stories_dir2, s)))
            elif i <= 196961+12148:
                fvalid.write("%s\n" % (os.path.join(stories_dir2, s)))
            else:
                ftest.write("%s\n" % (os.path.join(stories_dir2, s)))
            i += 1

    for name in ["mapping_for_train.txt", "mapping_for_valid.txt", "mapping_for_test.txt"]:
        name = os.path.abspath(args.dataMap+name)
        classPath = '/share/home/bingxianren/40_torch-self-learning/stanford-corenlp-4.2.0/stanford-corenlp-4.2.0.jar' \
            if args.remote else 'I:\stanfordNlp\stanford-corenlp-4.2.0\stanford-corenlp-4.2.0.jar'

        command = ['java', '-cp', classPath, 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                   '-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak',
                   'always', '-filelist', name, '-outputFormat', 'json', '-outputDirectory',
                   tokenized_stories_dir]

        subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    # os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(stories1) + len(stories2)
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir1 + "+" +stories_dir2, num_orig))
    print("Successfully finished tokenizing %s and %s to %s.\n" % (stories_dir1, stories_dir1, tokenized_stories_dir))

def format_to_bert(args, log):
    if (args.dataset != ''):
        datasets = [args.dataset]
    else:
        datasets = ['test', 'train', 'valid']
    for corpus_type in datasets:
        a_lst = []
        for json_f in glob.glob(pjoin(args.dataJson, '*.' + corpus_type + '.*.json')):
            real_name = json_f.split('\\')[-1]
            a_lst.append((json_f, args, pjoin(args.dataBert, real_name.replace('json', 'bert.pt')), log))
        # print(a_lst)
        if args.n_cpus >= 2:
            pool = Pool(args.n_cpus)
            for d in pool.imap(_format_to_bert, a_lst):  # imap(func, iterable, chunksize=0)
                pass

            pool.close()
            pool.join()
        else:
            for para in a_lst:
                _format_to_bert(para)

def _format_to_bert(params):
    json_file, args, save_file, log = params
    if (os.path.exists(save_file)):
        logger.info('Ignore %s' % save_file)
        return

    bert = BertData(args)

    logger.info('Processing %s' % json_file)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        if args.oracle_mode == 'greedy':
            oracle_ids, totalScore = greedy_selection(source, tgt, 3, args)  # Todo oracle ids and labels
        elif args.oracle_mode == 'combination':
            oracle_ids = combination_selection(source, tgt, 3)
        try:
            b_data = bert.preprocess(source, tgt, oracle_ids)
            if b_data is None:
                continue
            indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt, newOracle = b_data
            b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids,
                           'clss': cls_ids, 'src_txt': src_txt, "tgt_txt": tgt_txt,
                           "oracle_ids": newOracle, "totalScore": totalScore,
                           "desc":"clss_sents{} -->label_sents{}|{}".format(len(cls_ids), len(labels), len(source)),
                           "totalLen":len(segments_ids)}
            if len(newOracle) >0: datasets.append(b_data_dict)
        except:
            log.info(str(json_file) + " has bug!!!")
            print(str(json_file) + " has bug!!!")
            print(d)
    logger.info('Saving to %s' % save_file)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

def format_to_lines(args, log):
    corpus_mapping = {}
    for corpus_type in ['train','valid', 'test']:   # 'valid', 'test'
        temp = []
        for line in open(pjoin(args.dataMap, 'mapping_for_' + corpus_type + '.txt')):
            # temp.append(hashhex(line.strip()))
            # temp.append(line.strip()) # line.strip().split("\\")[-1].split(".")[0]
            temp.append(line.strip().split("\\")[-1].split(".")[0])  # line.strip().split("\\")[-1].split(".")[0]

        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    # for f in glob.glob(pjoin(args.raw_path, '*.json')):
    for f in glob.glob(pjoin(args.dataToken, '*.json')):
        real_name = f.split('\\')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)

    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.dataJson + "cnndm", corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}

# 以下两个函数没有用
def calScore(cans, abstract):
    # 函数功能：计算最优的摘要组合以及分数->计算某一个摘要和标准摘要的分数
    # cans: 一维列表
    # abstract标准摘要句融合为一维列表

    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    abstract = _rouge_clean(' '.join(abstract)).split()
    # 原文 二维列表  不需要了
    sents = _rouge_clean(' '.join(cans)).split()

    # _get_word_ngrams 的第二个参数是 二维列表
    evaluated_1grams = _get_word_ngrams(1, [sents])
    reference_1grams = _get_word_ngrams(1, [abstract])
    evaluated_2grams = _get_word_ngrams(2, [sents])
    reference_2grams = _get_word_ngrams(2, [abstract])

    candidates_1 = set.union(*map(set, evaluated_1grams))
    candidates_2 = set.union(*map(set, evaluated_2grams))
    rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
    rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
    rouge_score = rouge_1 + rouge_2

    return rouge_score

def get_mmr_regression(source, tgt, oracle):
    # 函数功能：计算每一步的分数增益，其实生成oracle的时候就可以计算了

    # src\tgt是二维列表，每一个元素是一个句子单词构成的列表
    source = [" ".join(s) for s in source]  # 一维列表
    tgt = [" ".join(s) for s in tgt]  # 一维列表
    selected = []
    selected_id = []
    prev_rouge = 0
    res_buf = []
    for sent_id in oracle:
        # 每一步 的候选范围 已经选择的selected + 所有句子之一
        # candidate是二维列表，x是一句话对应的字符串
        candidates = [(selected + [x]) for x in source]

        # 计算每一个候选的rouge分数
        cur_rouge = [calScore(can, tgt) for can in candidates]

        # 已经选择的句子内容
        selected.append(source[sent_id])
        selected_id.append(sent_id)

        # 计算分数增益
        out_rouge = [(x - prev_rouge) for x in cur_rouge]
        res_buf.append(out_rouge)

        prev_rouge = max(cur_rouge)

    return res_buf

# 代码主要修改的地方：
# 添加 train、test、valid外层循环
# 对数据增加 totalSteps用于生成打标
# 将oracle与labels对齐
# 这个oracle是截断后的，如果要计算oracle得分，最好重新生成一份数据：不要对原文截断

