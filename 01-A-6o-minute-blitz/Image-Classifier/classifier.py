from myUtils import *
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

train_bs = 4
lr = 1e-3
max_epoch = 5

# 数据
train_txt_path = 'Data/train.txt'
classes_name = ['plane', 'car', 'bird']

normMean = [0.4948052, 0.48568845, 0.44682794]  # 通道均值
normStd  = [0.24580306, 0.24236229, 0.2603115]  # 通道方差
# test参数
# normMean_test = [0.44871908, 0.48598012, 0.4943982]
# normStd_test = [0.25957936, 0.23998784, 0.24234861]

normTransform = transforms.Normalize(mean=normMean, std=normStd)
# normTransform_test = transforms.Normalize(mean=normMean_test, std=normStd_test)

# 数据预处理设置
train_transtorm = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset2(txt_path=train_txt_path, transform=train_transtorm)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)

# 定义网络
this_net = Net()
# this_net.zero_grad()
# this_net.initialize_weights()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=this_net.parameters(), lr=lr, momentum=0.9)

# 训练
for epoch in range(max_epoch):
    for index, data in enumerate(train_loader):
        data_x, label = data
        optimizer.zero_grad()
        output = this_net(data_x)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if (index+1) % 5 == 0:
            # 左对齐， 宽度为5，小数位数2
            print('Training: Epoch[{:2}/{:2}] Loss:{:<5.2f}'.
            format(epoch+1, max_epoch, loss.item()))

