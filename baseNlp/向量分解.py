import torch

def depose(vec1, vec2):
    # batch, seqLen, hiddenSize
    aStandard = vec1 / torch.norm(vec1)  # batch, seqLen, hiddenSize # a方向上的单位向量
    bParaLen = torch.matmul(aStandard, vec2.transpose(2, 1))  # 投影长度
    para = aStandard * bParaLen
    vert = vec2 - para
    return para, vert

def depose2(vec1, vec2):
    # batch, seqLen, hiddenSize
    aStandard = vec1 / torch.norm(vec1)  # batch, seqLen, hiddenSize # a方向上的单位向量
    bParaLen = (aStandard * vec2).sum(-1).unsqueeze(-1)
    para = aStandard * bParaLen
    vert = vec2 - para
    return para, vert

q = torch.randn(2, 4, 768)
v = torch.randn(2, 4, 768)
para, vert = depose2(q, v)
print(para.shape)
print(vert.shape)
