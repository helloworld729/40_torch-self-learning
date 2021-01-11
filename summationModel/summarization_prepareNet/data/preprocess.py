''' Handling the data io '''
import argparse
import torch
import Constants as Constants

def isNumOrChar(word):
    if "0"<= word <= "9" or "a" <= word <= "z" or "A" <= word <= "Z":
        return True
    return False

def isChinese(word):
    if '\u4e00' <= word <= '\u9fff':
        return True
    return False

def isPunctuation(word):  # 中文单引号，英文 引号、问号、冒号 左右括号替换
    if word in ["，", "。", "“", "”", "：", "；", "、", "？", "！", "《", "》", "（", "）", ".", "%", "…", "-", "—", "#"]:
        return True
    return False

def replace_word(sen):
    if not sen: return sen
    sen = sen.replace("‘","“")
    sen = sen.replace("’", "”")
    sen = sen.replace("'","“")
    sen = sen.replace("'", "”")
    sen = sen.replace("‘","“")
    sen = sen.replace("(","（")
    sen = sen.replace(")", "）")
    sen = sen.replace(":","：")
    return sen

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    # 函数功能：读取文本
    # 输入：文件地址、长度上限设定、是否保留大写
    # 输出：列表，每个元素对应一句话，如果非空，元素内容为 [开始标志]+句子中单词列表+[结束标志]

    # 总的结果列表
    word_insts = []

    # 统计有多少句子被修剪过
    trimmed_sent_count = 0
    with open(inst_file, encoding='utf-8') as f:
        for sent in f:
            # 数据预处理的时候自己增加了回车符号：\n
            sent = replace_word(sent)
            words = [i for i in sent]
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    # 函数功能：构建 world:index字典
    # 输入：read_instances_from_file的返回结果
    # 输出：world:index字典

    # 所有token的集合
    def isNeed(word):
        return isChinese(word) or isNumOrChar(word) or isPunctuation(word)
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] 原始词表容量 =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    # 统计单词出现的次数
    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    # 将出现次数大于下限的单词添加到词典中
    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if isNeed(word) and count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] 修剪之后的词表规模:= {},'.format(len(word2idx)),
          '每个单词至少出现过:= {}'.format(min_word_count))
    print("[Info] 忽略的单词数目:= {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    # 功能：将实际的单词列表 借助word2idx字典 转化为 index列表
    # 输入：单字实例列表， word2idx字典
    # 输出：句子的index序列化之后的 二维列表
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', default='train.former.txt')
    parser.add_argument('-train_tgt', default='train.latter.txt')
    parser.add_argument('-vocabUseValidation', default=True)
    parser.add_argument('-has_validation', default=True)
    parser.add_argument('-valid_src', default='valid.former.txt')
    parser.add_argument('-valid_tgt', default='valid.latter.txt')
    parser.add_argument('-save_data', default='save_file/file_saved.txt')
    # 变量 注释 参数, 好像只能用--后面的参数
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=150)
    parser.add_argument('-min_word_count', type=int, default=10)
    # parser.add_argument('-keep_case', action='store_true')  # 是否保持小写
    # parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-share_vocab', default=True)
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    # 设置句子长度上限值
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, False)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, False)

    if len(train_src_word_insts) != len(train_tgt_word_insts):
        print('[Warning] 训练集两侧数据不相等.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]

    # 只有输入输出全都不是空的时候，才添加到数据集中
    train_src_word_insts, train_tgt_word_insts = list(zip(*[
        (s, t) for s, t in zip(train_src_word_insts, train_tgt_word_insts) if s and t]))

    # Validation set
    if opt.has_validation:
        valid_src_word_insts = read_instances_from_file(
            opt.valid_src, opt.max_word_seq_len, False)
        valid_tgt_word_insts = read_instances_from_file(
            opt.valid_tgt, opt.max_word_seq_len, False)

        if len(valid_src_word_insts) != len(valid_tgt_word_insts):
            print('[Warning] 验证集两侧数据不相等.')
            min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts))
            valid_src_word_insts = valid_src_word_insts[:min_inst_count]
            valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]

        #- Remove empty instances
        valid_src_word_insts, valid_tgt_word_insts = list(zip(*[
            (s, t) for s, t in zip(valid_src_word_insts, valid_tgt_word_insts) if s and t]))

    # 本地是够是否已经有字典
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        # 翻译两侧的word2idx字典是否 分开
        if opt.share_vocab:
            print('[Info] 两侧的词表融合.开构建词表...')
            if opt.vocabUseValidation:
                insts = train_src_word_insts + train_tgt_word_insts + \
                        valid_src_word_insts + valid_tgt_word_insts
            else:
                insts = train_src_word_insts + train_tgt_word_insts
            word2idx = build_vocab_idx(insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] 两侧的词表不融合.')
            print('[Info] Build vocabulary for source.')
            if opt.vocabUseValidation:
                print('[Info] Build vocabulary for source.')
                src_word2idx = build_vocab_idx(train_src_word_insts + valid_src_word_insts, opt.min_word_count)
                print('[Info] Build vocabulary for target.')
                tgt_word2idx = build_vocab_idx(train_tgt_word_insts + valid_tgt_word_insts, opt.min_word_count)
            else:
                print('[Info] Build vocabulary for source.')
                src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
                print('[Info] Build vocabulary for target.')
                tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)


    # word to index
    print('[Info] 将源侧句子索引化.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx) if opt.has_validation else None

    print('[Info] 将终侧句子索引化.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx) if opt.has_validation else None

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] 数据保存', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()

