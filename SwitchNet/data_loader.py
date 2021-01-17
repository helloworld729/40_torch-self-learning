import json
# from config import CONFIG as conf
from config_pretrain import CONFIG as conf
from ast import literal_eval as make_tuple

#train_file = 'data/train.json'
#dev_file = 'data/dev.json'
#test_file = 'data/test.json'
train_file = conf['train_file']
dev_file = conf['dev_file']
test_file = conf['test_file']
max_sent_len = 120
max_doc_len = 80

def read_data(filename, add_first_sentence, keep_single_sent):
    # 句子sentence是列表， 一篇新闻all_sentence是二维列表
    # 整个训练集是三维列表，add_first_sentence是为每一篇新闻
    # 增加一个开头 <startsent>；keep_single_sent是指新闻
    # 长度<=1，也添加到数据集中。
    data = []  # 存储所有的训练数据，元素为每一篇新闻的数据
    with open(filename, encoding="utf-8") as in_file:
        # 每一行是一篇新闻
        for line in in_file:
            line = line.strip()
            # 存储所有的句子，元素为句子
            all_sentences = []
            if add_first_sentence:
                all_sentences = [['<startsent>']]
            count = len(all_sentences)
            for sentence in line.split('##SENT##'):
                #sentence = sentence.split()[:max_sent_len]
                # sentence是列表，元素为一句话对应的tokens
                sentence = sentence.split()
                if len(sentence) > 0:
                    all_sentences.append(sentence)
                count += 1
                if count == max_doc_len:
                    break
            if keep_single_sent or len(all_sentences) > 1:
                data.append(all_sentences)
    return data

#if __name__ == '__main__':
def get_train_dev_test_data(add_first_sentence = False, keep_single_sent=True,
                            ignore_train=False):
    train_data = None
    if not ignore_train:
        train_data = read_data(train_file, add_first_sentence, keep_single_sent)
    dev_data = read_data(dev_file, add_first_sentence, keep_single_sent)
    test_data = read_data(test_file, add_first_sentence, keep_single_sent)
    return train_data, dev_data, test_data
    #print(len(train_data))
    #print(train_data[0])

def read_oracle(file_name):
    # target是二维列表，oracle_tuple是一维列表，oracle内部是每篇文章
    # 对应的摘要句
    target = []
    with open(file_name, encoding="utf-8") as in_file:
        for line in in_file:
            oracle_tuple = make_tuple(line.split('\t')[0])
            if oracle_tuple is not None:
                oracle_tuple = list(oracle_tuple)
                new_oracle_tuple = [i for i in oracle_tuple if i < max_doc_len]
                oracle_tuple = new_oracle_tuple
            else:
                oracle_tuple = []
            target.append(oracle_tuple)
    return target

def read_target_txt(file_name, is_combine=True):
    # target 是二维列表，内部是目标摘要构成的
    # 的一维列表sentence，sentence的元素是句子字符串
    target = []
    with open(file_name, encoding="UTF-8") as in_file:
        for line in in_file:
            line = line.strip()
            sentences = line.split('##SENT##')
            if is_combine:
                target.append('\n'.join(sentences))
            else:
                target.append(sentences)
    return target

def read_target_20_news(file_name):
    target = []
    with open(file_name, encoding="UTF-8") as in_file:
        for line in in_file:
            line = line.strip()
            target.append(int(line))
    return target
