''' This module will handle the text generation with beam search. '''
import torch
import torch.nn.functional as F
from transformer.Beam import Beam
from transformer.Models import Transformer

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt.cuda else 'cpu')

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt

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

        model.load_state_dict(checkpoint['model'])  # module load
        print('[Info] Trained model state loaded.')

        # model.word_prob_prj = nn.LogSoftmax(dim=1)  # 类可以这样增加属性?

        model = model.to(self.device)

        self.model = model
        self.model.eval()

    def translate_batch(self, src_seq, src_pos):  # seq_id, seq_pos
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''  # 句子中某个索引的位置记忆
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            """
            Sentences which are still active are collected, so the decoder will not run on completed sentences.
            :param src_seq:  编码id序列
            :param src_enc:  编码输出
            :param inst_idx_to_position_map:
            :param active_inst_idx_list: 激活的实例id序列
            :return:
            """
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, n_bm):
            """
            Decode and update beam status, and then return active beam idx
            :param inst_dec_beams:  beam object
            :param len_dec_seq:     当前解码的长度-->step
            :param src_seq:         复制后的id序列  batch*beam_size，seq_len
            :param enc_output:      复制后的编码输出序列  batch*beam_size，seq_len，d_model
            :param inst_idx_to_position_map:
            :param n_bm:            beam_size
            :return:
            """

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                """

                :param inst_dec_beams: 解码定义的 beam对象
                :param len_dec_seq:    当前的step 也就是解码的长度
                :return:
                """
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]  # 返回状态列表
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm):
                """
                解码预测结果
                :param dec_seq:  解码id序列
                :param dec_pos:  解码位置序列
                :param src_seq:  编码id序列 batch*beam_size，seq_len
                :param enc_output: 编码输出 batch*beam_size，seq_len，d_model
                :param n_active_inst:
                :param n_bm:
                :return: batch，beam_size，vocab_size 每一个batch激活序列的预测分布
                """
                dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)  # 解码函数 batch*beam_size,1,512
                dec_output = dec_output[:, -1, :]  # Pick the last step:   batch*beam_size, 512
                word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)  # batch*beam_size, 10(vacab_size)
                word_prob = word_prob.view(n_active_inst, n_bm, -1)  # 每一个batch的解码映射结果  batch*beam_size*vocab_size

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                """

                :param inst_beams:   beam对象
                :param word_prob:   batch，beam_size，vocab_size 每一个batch激活序列的预测分布
                :param inst_idx_to_position_map:
                :return:
                """
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():  # 遍历batch数据
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])  # 核心beam调用  返回是否遇到EOS
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)  # 实例个数-->待翻译的句子数目

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm)  # batch，beam_size，vocab_size 每一个batch激活序列的预测分布

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

# ########################################## step分割线 ################################################################
        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():  # 入口
            #-- Encode
            src_seq, src_pos = src_seq.to(self.device), src_pos.to(self.device)
            src_enc, *_ = self.model.encoder(src_seq, src_pos)  # batch, max_len, d_model

            #-- Repeat data for beam search
            n_bm = self.opt.beam_size
            n_inst, len_s, d_h = src_enc.size()  # 实例数目，句子长度，嵌入维度
            src_seq = src_seq.repeat(1, n_bm).view(n_inst * n_bm, len_s)  # batch*beam_size
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)  # batch, max_len*beam_size, d_model -->  batch*beam_size, max_len, d_model

            #-- Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]  # 定义beam搜索， 只有一个参数即 beam_size

            #-- Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))  # batch_size
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            #-- Decode
            for len_dec_seq in range(1, self.model_opt.max_token_seq_len + 1):  # 当前解码长度

                active_inst_idx_list = beam_decode_step(
                    inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, n_bm)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                    src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, self.opt.n_best)

        return batch_hyp, batch_scores
