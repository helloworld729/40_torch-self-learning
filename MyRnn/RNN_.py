import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch


class Rnn_single_direction(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=3):
        super(Rnn_single_direction, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                           num_layers=num_layers)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, h0=None):
        """
        前向传播过程
        :param input_data: (batch_size, seq_len, hidden_size)
        :param h0: (num_layers, batch_size, hidden_size)

        :param output: (batch, seq_len, hidden_size)
        :param hidden: (layers, batch, hidden_size)
        :param x: (batch_size, class_nums)

        :return: final output of net
        """
        output, hidden = self.rnn1(input_data, h0)
        index = self.rnn1.num_layers - 1
        x = F.relu(self.fc1(hidden[index]))  # [batch, 分类数目]
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.RNN):
                pass
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()


class Rnn_Bidirection(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=1, bidirectional=True):
        super(Rnn_Bidirection, self).__init__()
        self.rnn1 = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                           num_layers=num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, h0=None):
        """
        前向传播过程(将最后两个方向concat在一起的数据view切开再index索引)
        :param input_data: (batch_size, seq_len, hidden_size)
        :param h0: (num_layers * 2, batch_size, hidden_size)
        :param output: (batch, seq_len, 2*hidden_size)
        :param hidden: (layers*2, batch, hidden_size)
        :param hid: (layers*directions, batch, hidden_size-->（layers，2, batch, hidden_size）
        :param hid_right: (layers, batch_size, hidden_size)
        :param hid_left: (layers, batch_size, hidden_size)
        :param hr_last: (batch_size, hidden_size)
        :param x: (batch_size, class_nums)

        :return: final output of net
        """

        # h0 = torch.randn(size=(self.rnn1.num_layers * 2, 4, self.rnn1.hidden_size))
        output, hidden = self.rnn1(input_data, h0)
        hid = hidden.view(self.rnn1.num_layers, 2, 4, self.rnn1.hidden_size)
        index = torch.tensor([0], dtype=torch.long)
        hid_right = torch.index_select(hid, dim=1, index=index).squeeze(1)  # layers, batch, hidden_size
        hid_left = torch.index_select(hid, dim=1, index=index + 1).squeeze(1)  # layers, batch, hidden_size
        hr_last = hid_right[self.rnn1.num_layers - 1]  # last_layer
        x = F.relu(self.fc1(hr_last))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.RNN):  # 因为各层的权重和bias不共享，所以会有很多的参数需要初始化，可以用方法_all_weights获取所有的参数名
                # weight_ih_l0 = m._all_weights[0][0]  # 第一个索引确定层与方向
                # shape = m._parameters[weight_ih_l0].size()
                # m._parameters[weight_ih_l0] = torch.zeros(size=shape)
                pass

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()


class GRU_single_direction(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=3):
        super(GRU_single_direction, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                           num_layers=num_layers)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, h0=None):
        """
        前向传播过程

        """
        output, hidden = self.gru1(input_data, h0)
        index = self.gru1.num_layers - 1
        x = F.relu(self.fc1(hidden[index]))  # [batch, 分类数目]
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                pass
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()


class GRU_Bidirection(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=1, bidirectional=True):
        super(GRU_Bidirection, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                           num_layers=num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, h0=None):
        """
        前向传播过程(将最后两个方向concat在一起的数据view切开再index索引)

        :return: final output of net
        """

        # h0 = torch.randn(size=(self.rnn1.num_layers * 2, 4, self.rnn1.hidden_size))
        output, hidden = self.gru1(input_data, h0)
        hid = hidden.view(self.gru1.num_layers, 2, 4, self.gru1.hidden_size)
        index = torch.tensor([0], dtype=torch.long)
        hid_right = torch.index_select(hid, dim=1, index=index).squeeze(1)  # layers, batch, hidden_size
        hid_left = torch.index_select(hid, dim=1, index=index + 1).squeeze(1)  # layers, batch, hidden_size
        hr_last = hid_right[self.gru1.num_layers - 1]  # last_layer
        x = F.relu(self.fc1(hr_last))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):  # 因为各层的权重和bias不共享，所以会有很多的参数需要初始化，可以用方法_all_weights获取所有的参数名
                # weight_ih_l0 = m._all_weights[0][0]  # 第一个索引确定层与方向
                # shape = m._parameters[weight_ih_l0].size()
                # m._parameters[weight_ih_l0] = torch.zeros(size=shape)
                pass

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()


class LSTM_single_direction(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=3):
        super(LSTM_single_direction, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                            num_layers=num_layers, dropout=0, bias=True)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, hid):
        """
        前向传播
        :param input_data: (batch_size, seq_len, input_size)
        :param hid: [h0, c0]
        :param h0: (num_layers , batch_size, hidden_size)
        :param c0: (num_layers , batch_size, hidden_size)
        :param output: (batch, seq_len, hidden_size)
        :param hidden: [hn, cn]
        :param hn: (num_layers, batch, hidden_size)
        :param cn: (num_layers, batch, hidden_size)
        :return x:(batch_size, class_nums)
        """
        hx = hid if isinstance(hid, tuple) else None
        output, hidden = self.lstm1(input=input_data, hx=hx)
        hn, cn = hidden
        index = self.lstm1.num_layers - 1
        x = F.relu(self.fc1(hn[index]))  # [batch, 分类数目]
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                pass
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()


class LSTM_Bidirection(nn.Module):
    def __init__(self, input_size=10, hidden_size=15, batch_first=True, num_layers=1, bidirectional=True):
        super(LSTM_Bidirection, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first,
                           num_layers=num_layers, bidirectional=bidirectional)
        self.fc1 = nn.Linear(15, 3)  # 3分类

    def forward(self, input_data, hid=None):
        """
        前向传播
        :param input_data: (batch_size, seq_len, input_size)
        :param hid: [h0, c0]
        :param h0: (num_layers*2 , batch_size, hidden_size)
        :param c0: (num_layers*2 , batch_size, hidden_size)
        :param output: (batch, seq_len, hidden_size*2)
        :param hidden: [hn, cn]
        :param hn: (num_layers*2, batch, hidden_size)
        :param cn: (num_layers*2, batch, hidden_size)
        :return x:(batch_size, class_nums)
        """
        # h0 = torch.randn(size=(self.rnn1.num_layers * 2, 4, self.rnn1.hidden_size))
        output, hidden = self.lstm1(input_data, hid)
        hn, cn = hidden
        hid = hn.view(self.lstm1.num_layers, 2, 4, self.lstm1.hidden_size)
        index = torch.tensor([0], dtype=torch.long)
        hid_right = torch.index_select(hid, dim=1, index=index).squeeze(1)     # layers, batch, hidden_size
        hid_left = torch.index_select(hid, dim=1, index=index + 1).squeeze(1)  # layers, batch, hidden_size
        hr_last = hid_right[self.lstm1.num_layers - 1]  # last_layer
        x = F.relu(self.fc1(hr_last))
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):  # 因为各层的权重和bias不共享，所以会有很多的参数需要初始化，可以用方法_all_weights获取所有的参数名
                # weight_ih_l0 = m._all_weights[0][0]  # 第一个索引确定层与方向
                # shape = m._parameters[weight_ih_l0].size()
                # m._parameters[weight_ih_l0] = torch.zeros(size=shape)
                pass

            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std=0.01)
                m.bias.data.zero_()