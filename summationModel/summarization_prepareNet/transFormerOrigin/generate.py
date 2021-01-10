''' Translate input text with trained model. '''
from tqdm import tqdm
import torch, torch.utils.data, argparse, os
from dataset import collate_fn, TranslationDataset
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

def main(author=False):
    if author:from transformer.Translator import Translator
    else:from transformer.My_Translator import Translator
    parser = argparse.ArgumentParser(description='generate.py')
    # parser.add_argument('-model', default='weights/transformer_accu_38.785.chkpt',
    #                     help='Path to model .pt file')
    parser.add_argument('-src', default='data/test.formmer',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', default='data/save_file/file_saved.txt',
                        help='Source sequence to decode (one line per sequence)')
    # parser.add_argument('-output', default='data/pred2.txt',  # 翻译结果保存文件路径
    #                     help="""Path to output the predictions (each line will
    #                     be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=2,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = True if torch.cuda.is_available() else False

    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)

    # 保存有长度数据，文件位置信息
    preprocess_settings = preprocess_data['settings']
    max_trans_len = preprocess_settings.max_word_seq_len

    # 原始数据list
    test_src_word_insts = read_instances_from_file(
        opt.src,
        max_trans_len,
        preprocess_settings.keep_case)
    # 将句子序列转化为
    test_src_insts = convert_instance_to_idx_seq(
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    # 遍历所有的模型文件
    for check_point in os.listdir("weights/"):
        opt.model = "weights/" + check_point
        model_name = "".join(opt.model.split("_")[2:])
        # 设定对应的输出目录
        opt.output = 'data/gene_poems/'+ model_name + ".txt"
        # 加载模型
        translator = Translator(opt)

        with open(opt.output, 'w', encoding='utf-8') as f:
            # leave设置为False，是的在一行持续输出
            for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
                # batch是批量数据(单侧待翻译数据index列表) + 位置索引, *使得参数列表化
                # (列表有两部分内容，1是seq的id序列，2是对应的未知序列)
                all_hyp, all_scores = translator.translate_batch(*batch)
                for i in range(len(all_hyp)):
                    idx_seqs, eos = all_hyp[i]["poem_idx"], all_hyp[i]["EOS"]
                    idx2word = test_loader.dataset.tgt_idx2word
                    if not author: pred_line = ''.join([idx2word[idx] for idx in idx_seqs])
                    else:          pred_line = ''.join([idx2word[idx] for idx in idx_seqs[0]][0])
                    formmer = ''.join([idx2word[idx] for idx in batch[0][i].data.numpy()]).\
                        replace("\n", "").replace("<s>", "").replace("</s>", "").replace("<blank>", "")
                    latter = pred_line.replace("\n", "").replace("<s>", "") + "。"
                    if eos: latter += "selfEnd"
                    f.write(formmer + "--->" + latter + "\n\n")
        print('[Info] Finished.')

if __name__ == "__main__":
    main()

