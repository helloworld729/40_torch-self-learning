from math import log
from numpy import array

"""
三层for结构：
1、遍历每一个step
2、遍历beam记忆
3、遍历当前step的每一个元素

如果是NLP中的时s2s模型，应该是：
1、遍历batch数据
2、遍历beam记忆
3、遍历当前step的每一个元素
"""

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]  # 最优结果集初始化  seq列表+probility在封装成list
    # walk over each step in sequence
    for row in data:  # 一行数据为一个step，遍历step
        all_candidates = list()  # 记录每一步中间计算过程
        # expand each current candidate
        for i in range(len(sequences)):  # 遍历batch_size
            seq, score = sequences[i]  # 到前一步的seq，到前一步的分数
            for j in range(len(row)):  # 遍历step内的的数据
                candidate = [seq + [j], score * -log(row[j])]  # 当前序列对应的候选集（1变多的过程）
                all_candidates.append(candidate)

        ordered = sorted(all_candidates, key=lambda tup: tup[1])  # 列表排序
        sequences = ordered[:k]  # 取出前k个对数较小，即概率较大的
    return sequences

# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],  # 第一个step
        [0.5, 0.4, 0.3, 0.2, 0.1],  # 第2个step
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)

# decode sequence
result = beam_search_decoder(data, 3)
for seq in result:
    print(seq)