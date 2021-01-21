import torch
import traceback
import sys
from torch.utils.data import Dataset

def train_collete(batch):
    input_token_ids_list = []
    segment_ids_list = []
    input_mask_list = []
    label_list = []
    max_len = 0
    for sample in batch:
        if max_len<len(sample[0]):
            max_len = len(sample[0])
    for sample in batch:
        padding_length = max_len-len(sample[0])
        input_token_ids_list.append(sample[0]+([0]*padding_length))
        segment_ids_list.append(sample[1]+[1]*padding_length)
        input_mask_list.append(sample[2]+[0]*padding_length)
        label_list.append(sample[3])
    return torch.LongTensor(input_token_ids_list),torch.LongTensor(segment_ids_list),torch.LongTensor(input_mask_list),torch.LongTensor(label_list)

def train_collete2(batch):
    """设置了长度上限"""
    input_token_ids_list = []
    segment_ids_list = []
    input_mask_list = []
    label_list = []
    max_len = 0
    for sample in batch:
        if max_len<len(sample[0]):
            max_len = len(sample[0])
    max_len = min(max_len, 150)
    for sample in batch:  # input_token_ids, segment_ids, input_mask, label
        padding_length = max_len-len(sample[0])
        if padding_length >= 0:
            input_token_ids_list.append(sample[0]+([0]*padding_length))
            segment_ids_list.append(sample[1]+[1]*padding_length)
            input_mask_list.append(sample[2]+[0]*padding_length)
        else:
            input_token_ids_list = input_token_ids_list[:padding_length]
            segment_ids_list = segment_ids_list[:padding_length]
            input_mask_list = input_mask_list[:padding_length]
        label_list.append(sample[3])
    return torch.tensor(input_token_ids_list),torch.tensor(segment_ids_list),torch.tensor(input_mask_list),torch.tensor(label_list)

def fun(f):
    lst = f.read()
    lst = lst.replace("\n", "")
    lst = [w for w in lst][1:]
    return lst

class news_dataset(Dataset):
    def __init__(self, token, is_training):  # 阅读csv文件的返回结果， 切词器， 长度上限
        self.token = token  # 切词器
        self.data = fun(open("data/train.txt", encoding="utf-8")) # list
        self.len = len(self.data)  # 数据集长度
        self.is_training = is_training

    def __getitem__(self, item):
        item_data = self.data[item]
        # token = self.token.tokenize(item_data)
        token = item_data

        tokens = ["[CLS]"]+[token]+["[SEP]"]
        segment_ids = [0, 0, 0]
        input_mask = [1]*len(tokens)  # 为了在collect函数中辅助生成pad
        input_token_ids = self.token.convert_tokens_to_ids(tokens)
        if self.is_training:
            # label = int(item_data['label'])  # int64 label
            label = self.token.vocab.get(token, self.token.vocab.get(self.token.unk_token))
            return input_token_ids, segment_ids, input_mask, label
        else:
            qid = int(item_data['qid'])  # int64 label
            return input_token_ids, segment_ids, input_mask, qid

    def __len__(self):
        return self.len



