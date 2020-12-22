import random
import torch.nn as nn
from torch import optim
from NlpFromScratch_2.seq2seq_utils import *
from NlpFromScratch_2.seq2seq_models import Lang, EncoderRNN, AttnDecoderRNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ############################ 超参数 ######################################
n_iters = 45
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
hidden_size = 32
teacher_forcing_ratio = 0.5

# ############################ 数据 ########################################
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

# ############################ 网络 ########################################
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

# ############################ 训练 ########################################
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length=MAX_LENGTH):

    # 编码器隐向量初始化
    encoder_hidden = encoder.initHidden()

    # 优化器梯度清零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 输入的长度控制编码侧的时间步
    # 输出的长度控制解码侧的时间步
    input_length  = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # 初始化隐层输出
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # 初始化损失
    loss = 0

    # 序列编码
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            # input_tensor[ei]是一维Tensor，会在forward内部取出
            # 对应的embedding Tensor
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # 序列解码
    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # 两个分支的区别在于decoder_input的区别
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  # 输出对应的张量和索引
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total  = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)
                      for i in range(n_iters)]  # 每一句话是一个二维矩阵，两句话对应的Tensor构成 training_pairs的一个元素
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

trainIters(encoder1, attn_decoder1, n_iters, print_every=15)

