import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import Dataset
import torch

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf8') as f:
        words = set()  # 单词集合
        word_to_vec_map = {}  # 单词：向量 字典
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}  # 单词：索引 字典
        index_to_words = {}  # 索引：单词 字典
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename='data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])  # 第0列单元格
            emoji.append(row[1])   # 第1列单元



    return phrase, emoji

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


# "0": "\u2764\uFE0F"
emoji_dictionary = {"0": "\u2764",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
              
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def print_lost_curve(lost):
        plt.plot(lost)
        plt.show()
    

def plot_matrix(in_matrix):
    plt.matshow(in_matrix)
    plt.title('confusion_matrix')
    plt.colorbar()
    plt.ylabel('true')
    plt.xlabel('predict')
    plt.show()


def plot(data1_x, data1_y):
    """
    区间 标题 数据 标签 文本
    :param data1_x:
    :param data1_y:
    :return:
    """
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.title('{}'.format('beautifel_picture'))
    plt.scatter(data1_x.numpy(), data1_y.numpy(), c='r', label='class 0')
    # plt.scatter(data2_x.numpy(), data2_y.numpy(), c='b', label='class 1')
    plt.legend()
    plt.text(x=1.0, y=-1, s='haha', fontdict={'size':20, 'color':'red'})
    plt.show()


def model_evaluate(y, y_hat, plot_confusion=False):
    acc = 0
    if len(y) == len(y_hat):
        acc = metrics.precision_score(y, y_hat, average='macro')  # 微平均，精确率
        recall = metrics.recall_score(y, y_hat, average='micro')
        F1 = metrics.f1_score(y, y_hat, average='weighted')
        print('准确率：{}\n召回率：{}\nF1:{}'.format(acc, recall, F1))

        if plot_confusion:
            plot_matrix(confusion_matrix(y, y_hat))

    else:
        print('文件不等长')
    return acc


def predict(X, Y, W, b, word_to_vec_map):
    """
    Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
    
    Arguments:
    X -- input data containing sentences, numpy array of shape (m, None)
    Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
    
    Returns:
    pred -- numpy array of shape (m, 1) with your predictions
    """
    m = X.shape[0]
    pred = np.zeros((m, 1))
    
    for j in range(m):                       # Loop over training examples
        
        # Split jth test example (sentence) into list of lower case words
        words = X[j].lower().split()
        
        # Average words' vectors
        avg = np.zeros((50,))
        for w in words:
            avg += word_to_vec_map[w]
        avg = avg/len(words)

        # Forward propagation
        Z = np.dot(W, avg) + b
        A = softmax(Z)
        pred[j] = np.argmax(A)
        
    print("Accuracy: "  + str(np.mean((pred[:] == Y.reshape(Y.shape[0],1)[:]))))
    
    return pred


class MyDataset(Dataset):
    def __init__(self,):
        self.sentence = self.make_info_set()  # data_x和label构成的tupple -->list

    def __getitem__(self, index):
        data_x, label = self.sentence[index]
        return data_x, label

    def __len__(self):
        return len(self.sentence)

    @staticmethod
    def make_info_set():
        info_list = []
        x = torch.randn(100, 8, 10)
        label = np.random.randint(low=0, high=3, size=[100]).astype(np.int64)  # right open shape=(100,)

        for i in range(x.size(0)):
            data_x = x[i]  # 8*10
            y = label[i]   #
            rep = (data_x, y)
            info_list.append(rep)
        return info_list


def make_test(sen, vlove_vec):
    sentense = sen
    words_list = sentense.strip().lower().split()  # 词语列表
    word_array = []  # 一句话有很多单词构成，所有的词向量构成数组
    for word in words_list:
        vector = vlove_vec[word]
        word_array.append(vector)
    word_array = np.asarray(word_array)  # 转化为数组
    sentence_rep = np.mean(word_array, axis=0)
    sentence_rep = torch.tensor(sentence_rep)
    sentence_rep = torch.unsqueeze(sentence_rep, 0)
    sentence_rep = torch.tensor(sentence_rep, dtype=torch.float32)

    return sentence_rep



