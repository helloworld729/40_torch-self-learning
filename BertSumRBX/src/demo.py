import torch
import argparse
import numpy as np
from models.model_builder import newSummarizer as Summarizer
device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据

def getClsPos(text):
    cls = list()
    sep = list()
    for i in range(len(text)):
        if text[i] == "cls":
            cls.append(i)
        elif text[i] == "sep":
            sep.append(i)
    return cls, sep

def getSeg(cls, sep):
    seg = []
    for i in range(len(cls)):
        num = 1 if i & 1 else 0
        temp = [num] * (sep[i] - cls[i] + 1)
        seg.extend(temp)
    return seg


# 文本用空格隔开 # 输入、段标记、cls的位置，mask，mask_cls
txt1 = " cls a host of footballing legends graced the field to raise money for charity sep " \
      " cls bazilian ronaldo and zinedine zidane were among the stars taking part sep " \
      " cls clarence seedorf , fabian barthez and gianluca zambrotta also played sep " \
      " cls as did jay-jay okocha , david trezeguet and vladimir smicer sep "

txt2 = " cls it was the age of modernism , with an emphasis on open living spaces , and the landgate cottage in rye demonstrates this to a tee sep " \
       " cls travel through time with this airbnb rental , which hark back to the earthy aesthetic of the atomic age sep " \
       " cls the cottage , which sleeps four , features dark wood beamed ceilings and plenty of mid-century furnishings , situated ideally near the town citadel sep "

word = list(set([w for w in (txt1+txt2).split()]))
print(word)
txt1, txt2 = txt1.split(), txt2.split()
word2index, index2word = dict(), dict()
for i in range(len(word)):
    index2word[i+1] = word[i]
    word2index[word[i]] = i+1

# 输入token 的 index序列, cls的index != 0
x1 = [word2index[w] for w in txt1]  # 52
x2 = [word2index[w] for w in txt2]  # 75
maxTokenCounts = max(len(x1), len(x2))
x1 += [0] * (maxTokenCounts-len(x1))
x2 += [0] * (maxTokenCounts-len(x2))
# mask序列
mask1 = ~(torch.tensor(x1) == 0).reshape(1, -1)  # padding的位置为False
mask2 = ~(torch.tensor(x2) == 0).reshape(1, -1)  # padding的位置为False

# cls所处位置的序列(从0开始)
clss1, sep1 = getClsPos(txt1)  # 4句话
clss2, sep2 = getClsPos(txt2)  # 3句话
# segmegnt段序列
segment1 = getSeg(clss1, sep1)
segment2 = getSeg(clss2, sep2)
maxSegmentCounts = max(len(segment1), len(segment2))
segment1 += [0] * (maxSegmentCounts - len(segment1))
segment2 += [0] * (maxSegmentCounts - len(segment2))
segment1 = torch.tensor(segment1).reshape(1, -1)
segment2 = torch.tensor(segment2).reshape(1, -1)

maxSenCounts = max(len(clss1), len(clss2))
clss1 += [-1]*(maxSenCounts - len(clss1))  # 根据最长句子数来
clss2 += [-1]*(maxSenCounts - len(clss2))  # 根据最长句子数来
maskCls1 = ~(torch.tensor(clss1) == -1).reshape(1, -1)  # padding的位置为False
maskCls2 = ~(torch.tensor(clss2) == -1).reshape(1, -1)  # padding的位置为False
# mask_cls序列
clss1, clss2 = torch.tensor(clss1), torch.tensor(clss2)
clss1[clss1 == -1] = 0
clss2[clss2 == -1] = 0
clss1, clss2 = clss1.unsqueeze(0), clss2.unsqueeze(0)

embeddings = np.random.randn(len(word)+5, 768)  # 构建30维的词向量

x1 = torch.from_numpy(np.asarray([embeddings[x] for x in x1]), ).unsqueeze(0)
x2 = torch.from_numpy(np.asarray([embeddings[x] for x in x2])).unsqueeze(0)
x = torch.cat((x1, x2), dim=0)  # 2, seqLen, 30
segs = torch.cat([segment1, segment2], dim=0)
clss = torch.cat((clss1, clss2), dim=0)
mask = torch.cat((mask1, mask2), dim=0)
maskCls = torch.cat((maskCls1, maskCls2), dim=0)

print("maxLen:", maxTokenCounts)
print("x1.shape:\n", x1.shape)
print("x2.shape:\n", x2.shape)
print("x.shape\n", x.shape)
print("seg:\n", segs)
print("clss\n", clss)
print("mask\n", mask)
print("maskCls:\n", maskCls)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-encoder", default='transformer', type=str, choices=[
                                  'classifier','transformer','rnn','baseline'])
    parser.add_argument("-mode", default='validate', type=str, choices=[
                                'train', 'validate', 'test', 'lead', 'oracle'])

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
    parser.add_argument("-max_grad_norm", default=0.1, type=float,
                        help="当默认!=0时，可以启动梯度修剪")

    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-seed', default=527, type=int)

    # 参数
    args = parser.parse_args()

    # 模型
    summarizer = Summarizer(args, device)

    # 输入、段标记、cls的位置，mask，mask_cls
    assert x.shape    == (2, 75, 768)
    assert segs.shape == (2, 75)
    assert clss.shape == (2, 4)
    assert mask.shape == (2, 75)
    assert maskCls.shape == (2, 4)
    sent_scores, mask_cls = summarizer(x, segs, clss, mask, maskCls)
    print("hello")

    # 训练
    # summarizer()

