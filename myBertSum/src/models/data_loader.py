import gc
import glob
import random

import torch
import os
# os.chdir("/share/home/bingxianren/40_torch-self-learning/myBertSum")
# print(os.getcwd())
from myBertSum.src.others.logging import logger


class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None,  is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segs, 0))
            # mask = 1 - (src == 0)
            mask = ~(src == 0)
            clss = torch.tensor(self._pad(pre_clss, -1))
            # mask_cls = 1 - (clss == -1)
            mask_cls = ~(clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))

            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def load_dataset(args, corpus_type, shuffle):
    # 每一个语料库都有很多的数据，数据被组织成了不同的文件
    # 函数是一个生成器，每次返回一个文件中包含的数据
    # 这个数据是一个字典列表，字典中是每一篇文章的数据字典。

    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # 返回所有匹配的路径paths
    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    # new:一篇新闻的数据元组，count：当前minibatch中的新闻条数
    # 统计新闻文本的长度上限值
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents  = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))  # 最长news的长度
    # max_size = max(max_size, max_n_sents)
    # src_elements = count * max_size
    src_elements = count * max_n_sents
    return src_elements


class Dataloader(object):
    def __init__(self, args, filesets_generator,
                 device, shuffle, is_test):
        self.args = args
        self.filesets_generator = filesets_generator  # 面向语料的生成器，yield的是一个拉加载的文件的内容列表
        self.batch_size = args.batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        # filesets_generator是一个面向所有语料文件生成器,
        # _next函数首先指向其中的一个文件，即 self.cur_dataset
        # 然后基于 self.cur_dataset 构建buffer生成器
        self.cur_iter = self._next_dataset_iterator(filesets_generator)

        assert self.cur_iter is not None

    def __iter__(self):
        # datasets是语料库，d是文件，构成生成器
        dataset_iter = (d for d in self.filesets_generator)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch

            # 返回一个生成器
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        # 接受一个 生成器 作为参数
        # 首先，设置当前面向的文件即 self.cur_dataset
        # 然后，将当前文件的news构成的列表，送到DataIterator。
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            # cur_dataset对应的是一个文件对应的东西
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args=self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    # dataset是一个文件的news构成的列表
    def __init__(self, args, dataset,  batch_size,  device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def getDataFromDataset(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        # 取出字典的value
        src = ex['src']
        if('labels' in ex):
            labels = ex['labels']
        else:
            labels = ex['src_sent_labels']

        segs = ex['segs']
        if(not self.args.use_interval):
            segs=[0]*len(segs)
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        if(is_test):
            return src,labels,segs, clss, src_txt, tgt_txt
        else:
            return src,labels,segs, clss

    def batch_buffer(self, data, batch_size):
        # data: # 字典列表
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)  # src,labels,segs, clss
            if(ex is None):
                continue
            minibatch.append(ex)  # 数据元组
            # 统计当前时间的batch容量(以toke为单位)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1] if minibatch[:-1] else minibatch
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.getDataFromDataset()  # 字典列表
        for buffer in self.batch_buffer(data, self.batch_size):
            # 根据句子数排序, 生成排序列表
            buffer.sort(key=lambda x:len(x[3]))
            yield buffer

    def __iter__(self):  # 返回可迭代对象[生成器]，然后可以用for函数访问
        while True:
            for idx, minibatch in enumerate(self.create_batches()):
                # 将数据padding并且 转化为张量
                batch = Batch(minibatch, self.device, self.is_test)
                yield batch
            return


if 0:
    total_news = 0
    lastStep   = 0
    curStep    = 0
    def myLoadDataset(args, corpus_type, shuffle):
        # 每一个语料库都有很多的数据，数据被组织成了不同的文件
        # 函数是一个生成器，每次返回一个文件中包含的数据
        # 这个数据是一个字典列表，字典中是每一篇文章的数据字典。

        def _lazy_dataset_loader(pt_file, corpus_type):
            global lastStep
            global total_news
            dataset = torch.load(pt_file)
            print("步数：", curStep-lastStep)
            print("\n")
            print("文件名:{} 新闻数:{}".format(pt_file.split("\\")[-1], len(dataset)))

            total_news += len(dataset)
            lastStep = curStep
            return dataset

        pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)


    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch_size", default=6400, type=int)  # 1000
    parser.add_argument("-use_interval", default=True)
    parser.add_argument('-dataset', default='')
    parser.add_argument("-bert_data_path", default='../../bert_data/cnndm')

    args = parser.parse_args()

    datasets = myLoadDataset(args, "train", True)

    # 实例化迭代器
    dataLoader = Dataloader(args, datasets, "cpu", shuffle=False, is_test=False)

    for step, batch in enumerate(dataLoader):
        curStep = step
    print("batch_size:", args.batch_size)
    print("总步数:", curStep, "\n")
    print("新闻总数:", total_news, "\n")

# 问题：一个epoch有过少两个steps
# 每个文件有几个例子，有几个steps
# 文件名:cnndm.train.99.bert.pt 新闻数:2000
# batch_size: 6400
# 总步数: 23971
# 新闻总数: 287084