import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn.init as init


class LabelSmoothing(nn.Module):

    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        #self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing#if i=y的公式
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()#先深复制过来
        #print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))#otherwise的公式
        #print true_dist
        #变成one-hot编码，1表示按列填充，
        #target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class max_margin_loss(nn.Module):
    def __init__(self):
        super(max_margin_loss, self).__init__()
        pass
    def forward(self, labels, raw_logits, margin=0.4, downweight=0.5):
        # raw_logits = torch.norm(raw_logits,p=2,dim=,keepdim=True)
        labels = torch.zeros(labels.shape[0], 3).cuda().scatter_(1, labels.view(-1,1), 1)
        logits = raw_logits - 0.5
        # print(logits)
        # logits = raw_logits
        positive_cost = labels.type_as(raw_logits)*torch.pow(torch.max(torch.FloatTensor([0]).cuda(),margin-logits),2)
        negative_cost = (1-labels.type_as(raw_logits))*torch.pow(torch.max(torch.FloatTensor([0]).cuda(),logits+margin),2)
        # print(positive_cost)
        # print(negative_cost)
        return torch.sum((positive_cost*0.5+downweight*0.5*negative_cost),dim=-1).sum()


def squash(inputs, axis=2):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules  句子长度
    :param out_num_caps: number of capsules outputted from this layer  分类数
    :param out_dim_caps: dimension of output capsules  类别维度
    :param routings: number of iterations for the routing algorithm

    输出维度:batch，out_num, out_dim
    """
    def __init__(self,in_dim_caps, out_num_caps, out_dim_caps, routings=3):
        super(DenseCapsule, self).__init__()
        # self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.linear = nn.Linear(in_dim_caps, out_num_caps*out_dim_caps)

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        # x [B,L,H]
        # x_hat = [B,L,out_num_caps * out_dim_caps]
        # x = torch.stack([torch.cat([i,torch.cuda.FloatTensor([[0]*self.in_dim_caps]*(max_len-i.shape[0]))],) for i in x],dim=0)
        x_hat = self.linear(x).view(x.shape[0],x.shape[1],self.out_num_caps,self.out_dim_caps)  # batch, in_num, out_num, out_dim

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()
        # mask = (x.eq(0)[:,:,0]).unsqueeze(-1).repeat(1,1,self.out_num_caps)


        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = torch.zeros(x.shape[0], x.shape[1], self.out_num_caps).cuda()  # batch，in_num, out_num  就是论文中的b

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=2)
            # c.masked_fill_(mask,0)
            c = c.unsqueeze(-1)
            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c * x_hat, dim=1, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c * x_hat_detached, dim=1, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)
        return torch.norm(torch.squeeze(outputs, dim=1), p=2, dim=2)
