import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# import torch.optim as optim
from tool.transformers.modeling_bert import BertForSequenceClassification,BertConfig
from tool.transformers import AdamW
from tool.transformers import BertTokenizer
from my_data_loader import *
from torch.utils.data import DataLoader
import pandas as pd
import random
from sklearn import metrics
import argparse
# import numpy as np
from tqdm import tqdm
import torch
# ################################################ 参数 #############################################################
parser = argparse.ArgumentParser()  # 参数对象
parser.add_argument('--model_path',    help='模型路径',         default='./chinese_roberta/')
parser.add_argument('--log_file',      help='日志路径',         default='log/'     )
parser.add_argument('--weights',       help='权重文件',         default='weights/'          )
parser.add_argument('--learning_rate', help='学习率',           default=5e-5,   type=float)
# parser.add_argument('--adam_epsilon',  help='adam_epsilon',     default=0.009,    type=float)
# parser.add_argument('--weight_decay',  help='正则系数',         default=0.001,    type=float)
parser.add_argument('--max_seq_lens',  help='两个句子长度上限', default=100,      type=int)
parser.add_argument('--batch_size',    help='batch_size',       default=64,       type=int)
parser.add_argument('--MAX_EPOCH',     help='训练圈数',         default=10,       type=int)
parser.add_argument('--use_cuda',      help='use_cuda',         default=True,   type=bool)
parser.add_argument('--seed',          help='seed',             default=666,       type=int)
args = parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
set_seed(args)
print(args)
# ################################################ 数据 #############################################################
bert_token = BertTokenizer.from_pretrained(args.model_path, do_lower_case=False)  # 切词器

# data_dev =   pd.read_csv('data/dev_{}.csv'.format(i),  engine='c')  # 验证数据
train_dataset = news_dataset(bert_token, is_training=True)
# dev_dataset   = news_dataset(data_dev,   bert_token, args.max_seq_lens, is_training=True)  # 仅测试集时候为false
train_dataloader = DataLoader(train_dataset, shuffle=True,  batch_size=args.batch_size, collate_fn=train_collete2)  # 多进程可以用num_works
# dev_dataloader   = DataLoader(dev_dataset,   shuffle=False, batch_size=args.batch_size, collate_fn=train_collete)
# ################################################ 模型 #############################################################
config = BertConfig.from_pretrained(args.model_path, num_labels=21128)  # 0 1
model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)  # 模型加载
model = model.cuda() if args.use_cuda else model
if args.use_cuda:
    model = torch.nn.DataParallel(model)
model.train()
# ################################################ 优化 #############################################################
param_optimizer = list(model.parameters())  # 元组列表，元组为参数str名字， 参数值
optimizer = AdamW(param_optimizer, lr=args.learning_rate)  # 配置优化器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)  # 学习率下降策略
# ################################################ 日志 #############################################################
log_train_file = args.log_file + 'train.log'
log_tf = open(log_train_file, 'w')
# ################################################ 训练 #############################################################
log_steps = 100
j = 0
stop = False
for epoch in range(50):
    if not stop:
        F1_list = []; ACC_list = []; Loss_list = []; Recall_list = []
        model.train()
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
        for step, (input_token_ids, segment_ids, input_mask_list, label_list) in bar:
            optimizer.zero_grad()
            if args.use_cuda:
                model.cuda()
                input_token_ids = input_token_ids.cuda()
                segment_ids = segment_ids.cuda()
                input_mask_list = input_mask_list.cuda()
                label_list = label_list.cuda()
            # forward
            loss, logits = model(input_ids=input_token_ids, token_type_ids=segment_ids, attention_mask=input_mask_list, labels=label_list)
            # loss and backward
            loss.backward()
            # update
            optimizer.step()
            # scheduler.step()  # 更新学习率
            bar.set_description("loss {:.4f}".format(loss.cpu()))
            # ######################################## 指标 ##########################################
            _, predict = torch.max(logits, dim=1)
            label_list = label_list.data.cpu().numpy().squeeze()
            predict = predict.data.cpu().numpy().squeeze()
            acc = sum([i == j for i, j in zip(label_list, predict)]) / len(label_list)
            ACC_list.append(acc)
            Loss_list.append(loss.item())
            if (step + 1) % log_steps == 0:
                loss_avg   = sum(Loss_list)   / log_steps
                acc_avg    = sum(ACC_list)    / log_steps
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch + 1, args.MAX_EPOCH, step + 1, len(train_dataloader), loss_avg, acc_avg))
                log_tf.write('epoch:{epoch},loss:{loss: 8.5f},acc:{accu:3.3f}\n'.format(epoch=epoch, loss=loss_avg, accu=acc_avg))
                log_tf.flush()
                F1_list = []; ACC_list = []; Loss_list = []; Recall_list = []
                print("原文：", label_list)
                print("预测：", predict)
        # ########################################## 保存 ############################################
        net_save_path = args.weights + 'bert_params_{}.pkl'.format(j)
        j += 1
        # 状态字典是父类的函数
        if acc > 0.5:
            print('save model......')
            torch.save(model.state_dict(), net_save_path)  # 对象和保存路径
        if acc >=0.999999:
            stop = True
            print("Info: Finish")

