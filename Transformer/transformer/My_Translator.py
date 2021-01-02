''' This module will handle the text generation with beam search. '''
import torch
import copy
import torch.nn.functional as F
import transformer.Constants as Constants
from transformer.Models import Transformer

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt, max_trans_len):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt
        self.max_trans_len = max_trans_len

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            tgt_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_tgt_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout)

        model.load_state_dict(checkpoint['model'])  # 加载模型
        print('[Info] 模型加载完毕.')
        # model.word_prob_prj = nn.LogSoftmax(dim=1)  # 类可以这样增加属性?
        model = model.to(self.device)
        self.model = model
        self.model.eval()

    def init_transdict(self, enc_output):
        batch, *_ = enc_output.size()
        # id_seq(index序列，全部以BOS开头), score采用-log相乘所以初始化为1
        # beam翻译结果
        beam_item = {'id_sec': [2, ], 'pos_sec': [1, ], 'score': 1, 'EOS': False}
        # 句子翻译结果
        sen_item = {k: beam_item for k in range(self.opt.beam_size)}
        # 批量翻译结果
        trans_batch = {k: sen_item for k in range(batch)}
        return trans_batch

    def beam_search(self, enc_output, src_seq):
        """
        beam搜索step过程，输入编码侧的输出和原始的待编码序列输出翻译结果 循环过程为：
        1、初始化第一个解码输入为 BOS
        2、拿到解码结果 并存储到字典中
        3、构建新的解码输入
        4、终止条件 长度达到上限 或者遇到 EOS
        :param enc_output: batch, seq_len, d_model
        :param src_seq:    batch, seq_len
        """
        # 初始化 翻译字典(batch --> sentence --> beam)
        trans_batch_init = self.init_transdict(enc_output)  # 初始化搜寻结果
        step = 1
        while(step < self.max_trans_len):
            for sen_id, sen_trans in trans_batch_init.items():
                # 某个具体句子index序列
                this_sec = src_seq[sen_id].unsqueeze(0).to(self.device)
                # 某个具体句子编码序列
                this_enc = enc_output[sen_id].unsqueeze(0).to(self.device)
                temp_beam = []
                # for beam_id, beam_trans in sen_trans.items():   # 每一个beam  一变多
                for beam_id, beam_trans in trans_batch_init[sen_id].items():  # 每一个beam  一变多
                    # 如果还没有翻译出结束标志
                    if not beam_trans['EOS']:
                        # 当前翻译结果的id_sec索引序列，与pos_sec位置序列
                        dec_sec = torch.tensor(beam_trans['id_sec']) .view(1, -1)
                        dec_pos = torch.tensor(beam_trans['pos_sec']).view(1, -1)
                        # dec_output, *_ = self.model.decoder(dec_sec, dec_pos, this_seq, this_enc)
                        # 进行解码，输入this_sec是为了求mask，因为是针对每一个beam操作的，所以
                        # 返回的结果是 1,1,d_model
                        dec_output = self.model.decoder(dec_sec, dec_pos, this_sec, this_enc)
                        # dec_output shape: 1, d_model
                        dec_output = dec_output[:, -1, :]  # Pick the last step:   1, 512
                        # 映射为单词概率
                        word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1).float()
                        # topk结果：[0]表示word_prob最大的前beam_size个数值，[1]表示对应的索引
                        pred_list = word_prob.topk(self.opt.beam_size, dim=1, sorted=True)[1].squeeze()
                        score_list = word_prob.index_select(dim=1, index=pred_list).squeeze()  # 1 beam_size
                        for i in range(len(pred_list)):
                            temp_trans = copy.deepcopy(beam_trans)  # 同步改变了
                            if Constants.EOS == pred_list[i]:
                                temp_trans['EOS'] = True
                            else:
                                temp_trans['id_sec'].append(pred_list[i].item())
                                temp_trans['pos_sec'].append(temp_trans['pos_sec'][-1] + 1)
                                temp_trans['score'] += (score_list[i] * -1) if score_list[i] < 0 else score_list[i]  # 考位为0时候
                                # temp_trans['score'] = temp_trans['score']/len(temp_trans['pos_sec'])
                            temp_beam.append(temp_trans)
                        if step == 1:
                            break
                    # 该beam分支已经结束 直接添加
                    else:
                        temp_beam.append(beam_trans)

                # 步进完成后，统计结束数目
                finish_num = 0
                for beam_id, beam_trans in trans_batch_init[sen_id].items():
                    if beam_trans['EOS']:
                        finish_num += 1
                # 如果某一句话所有beam都已经结束，那么久跳过115行之后的操作过程
                # 直接进入下一句话
                if finish_num == len(trans_batch_init[sen_id]):
                    continue

                # 针对一句话
                print("句子{}对应的所有候选集在step{}".format(sen_id, step))
                ordered = sorted(temp_beam, key=lambda iner_dict: iner_dict['score'], reverse=False)  # 列表排序 顺序 小到大
                for item in ordered:
                    print(item)
                sequences = ordered[:self.opt.beam_size]  # 取出概率较大的  列表

                for item in ordered:
                    if item['EOS'] and (item not in sequences):
                        sequences.append(item)
                        print('增加自动终结序列...........................')

                sequences = {k: sequences[k] for k in range(len(sequences))}
                print("排序后的结果:")
                for value in sequences.values():
                    print(value)
                print("\n\n")
                # 替换
                trans_batch_init[sen_id] = sequences
            step += 1
        return trans_batch_init

    def translate_batch(self, src_seq, src_pos):  # 两个变量是编码侧的输入，扔进来一批要翻译的话
        with torch.no_grad():  # 入口
            # src_seq  # batch_size, seq_len(max)  2*......*30...
            # src_pos  # batch_size, seq_len(max)  1230...max_seq_len
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)

            # 加载模型，输出：batch_size, max_len, d_model
            src_enc, *_ = self.model.encoder(src_seq, src_pos)
            return_list = []
            # 返回一个batch的字典
            trans_result = self.beam_search(src_enc, src_seq)
            for sen_id, sen_candidate in trans_result.items():
                ordered = sorted(sen_candidate.items(), key=lambda x: x[1]['score'], reverse=False)  # 列表排序 顺序 小到大
                print("最终排序结果:")
                for item in ordered:
                    print(item)
                print("\n\n")
                # 每一句话对应得分最高的答案
                return_list.append(ordered[0][1]['id_sec'])
            return return_list, 0

# ######################################################################################################################
#     def beam_search(self, enc_output, src_seq):
#         """
#         beam搜索step过程，输入编码侧的输出和原始的待编码序列输出翻译结果 循环过程为：
#         1、初始化第一个解码输入为 BOS
#         2、拿到解码结果 并存储到字典中
#         3、构建新的解码输入
#         4、终止条件 长度达到上限 或者遇到 EOS
#         :param enc_output: batch, seq_len, d_model
#         :param src_seq:    batch, seq_len
#         """
#         # 初始化 翻译字典(batch --> sentence --> beam)
#         trans_batch_init = self.init_transdict(enc_output)  # 初始化搜寻结果
#         step = 1
#         while(step < self.max_trans_len):
#             for sen_id, sen_trans in trans_batch_init.items():
#                 # 某个具体句子index序列
#                 this_sec = src_seq[sen_id].unsqueeze(0).to(self.device)
#                 # 某个具体句子编码序列
#                 this_enc = enc_output[sen_id].unsqueeze(0).to(self.device)
#                 temp_beam = []
#                 # for beam_id, beam_trans in sen_trans.items():   # 每一个beam  一变多
#                 for beam_id, beam_trans in trans_batch_init[sen_id].items():  # 每一个beam  一变多
#                     # 如果还没有翻译出结束标志
#                     if not beam_trans['EOS']:
#                         # 当前翻译结果的id_sec索引序列，与pos_sec位置序列
#                         dec_sec = torch.tensor(beam_trans['id_sec']) .view(1, -1)
#                         dec_pos = torch.tensor(beam_trans['pos_sec']).view(1, -1)
#                         # dec_output, *_ = self.model.decoder(dec_sec, dec_pos, this_seq, this_enc)
#                         # 进行解码，输入this_sec是为了求mask，因为是针对每一个beam操作的，所以
#                         # 返回的结果是 1,1,d_model
#                         dec_output = self.model.decoder(dec_sec, dec_pos, this_sec, this_enc)
#                         # dec_output shape: 1, d_model
#                         dec_output = dec_output[:, -1, :]  # Pick the last step:   1, 512
#                         # 映射为单词概率
#                         word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1).float()
#                         # topk结果：[0]表示word_prob最大的前beam_size个数值，[1]表示对应的索引
#                         pred_list = word_prob.topk(self.opt.beam_size, dim=1, sorted=True)[1].squeeze()
#                         score_list = word_prob.index_select(dim=1, index=pred_list).squeeze()  # 1 beam_size
#                         for i in range(len(pred_list)):
#                             temp_trans = copy.deepcopy(beam_trans)  # 同步改变了
#                             if Constants.EOS == pred_list[i]:
#                                 temp_trans['EOS'] = True
#                             else:
#                                 temp_trans['id_sec'].append(pred_list[i].item())
#                                 temp_trans['pos_sec'].append(temp_trans['pos_sec'][-1] + 1)
#                                 temp_trans['score'] += (score_list[i] * -1) if score_list[i] < 0 else score_list[i]  # 考位为0时候
#                                 # temp_trans['score'] = temp_trans['score']/len(temp_trans['pos_sec'])
#                             temp_beam.append(temp_trans)
#                         if step == 1:
#                             break
#                     # 该beam分支已经结束 直接添加
#                     else:
#                         temp_beam.append(beam_trans)
#
#                 # 步进完成后，统计结束数目
#                 finish_num = 0
#                 for beam_id, beam_trans in trans_batch_init[sen_id].items():
#                     if beam_trans['EOS']:
#                         finish_num += 1
#                 # 如果某一句话所有beam都已经结束，那么久跳过115行之后的操作过程
#                 # 直接进入下一句话
#                 if finish_num == len(trans_batch_init[sen_id]):
#                     continue
#
#                 # 针对一句话
#                 print("句子{}对应的所有候选集在step{}".format(sen_id, step))
#                 ordered = sorted(temp_beam, key=lambda iner_dict: iner_dict['score'], reverse=False)  # 列表排序 顺序 小到大
#                 for item in ordered:
#                     print(item)
#                 sequences = ordered[:self.opt.beam_size]  # 取出概率较大的  列表
#
#                 for item in ordered:
#                     if item['EOS'] and (item not in sequences):
#                         sequences.append(item)
#                         print('增加自动终结序列...........................')
#
#                 sequences = {k: sequences[k] for k in range(len(sequences))}
#                 print("排序后的结果:")
#                 for value in sequences.values():
#                     print(value)
#                 print("\n\n")
#                 # 替换
#                 trans_batch_init[sen_id] = sequences
#             step += 1
#         return trans_batch_init
#
#     def translate_batch(self, src_seq, src_pos):  # 两个变量是编码侧的输入，扔进来一批要翻译的话
#         with torch.no_grad():  # 入口
#             # src_seq  # batch_size, seq_len(max)  2*......*30...
#             # src_pos  # batch_size, seq_len(max)  1230...max_seq_len
#             src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
#
#             # 加载模型，输出：batch_size, max_len, d_model
#             src_enc, *_ = self.model.encoder(src_seq, src_pos)
#             return_list = []
#
#             trans_result = self.beam_search(src_enc, src_seq)  # 字典
#             src_mask_len = src_seq.eq(0).sum(dim=1)
#             src_max_len = src_seq.size(1)
#             for sen_id, sen_candidate in trans_result.items():  # 赏罚规则
#                 sen_list = []
#                 for beam_id, beam_content in sen_candidate.items():
#                     # print(src_max_len,'hahah',src_mask_len[sen_id].item())
#                     src_len = src_max_len - src_mask_len[sen_id].item()  # 实际长度
#                     if 0.7*src_len < len(beam_content['pos_sec']) < 1.3*src_len:  # 长度适中
#                         beam_content['score'] *= 0.5
#                         if beam_content['EOS']:  # 奖赏自动终结
#                             beam_content['score'] *= 0.5
#
#                     elif 1.5*src_len < len(beam_content['pos_sec']):  # 太长
#                         beam_content['score'] *= (len(beam_content['pos_sec'])/src_len)
#                     elif 0.5*src_len > len(beam_content['pos_sec']):  # 太短
#                         beam_content['score'] *= (src_len / len(beam_content['pos_sec']))
#
#                     dec_set_len = set(beam_content['id_sec'])  # 集合长度
#                     if len(dec_set_len) > 0.1 * len(beam_content['id_sec']):  # 惩罚重复
#                         beam_content['score'] *= 2+(1 - len(dec_set_len)/len(beam_content['id_sec']) )
#
#                     sen_list.append(beam_content)
#
#                 ordered = sorted(sen_list, key=lambda iner_dict: iner_dict['score'], reverse=False)  # 列表排序 顺序 小到大
#
#                 print("最终排序结果:")
#                 for item in ordered:
#                     print(item)
#                 print("\n\n")
#
#                 return_list.append(ordered[0]['id_sec'])
#             return return_list, 0
# ######################################################################################################################

