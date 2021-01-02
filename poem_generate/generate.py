''' Translate input text with trained model. '''
from tqdm import tqdm
import torch, torch.utils.data, argparse
# from transformer.Translator import Translator
from transformer.My_Translator import Translator
from dataset import collate_fn, TranslationDataset
from preprocess import read_instances_from_file, convert_instance_to_idx_seq

def main(author=False):
    parser = argparse.ArgumentParser(description='generate.py')

    parser.add_argument('-model', default='weights/transformer.chkpt',
                        help='Path to model .pt file')
    parser.add_argument('-src', default='data/test.formmer',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', default='data/save_file/file_saved.txt',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='data/pred.txt',  # 翻译结果保存文件路径
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=2,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    # opt.cuda = not opt.no_cuda
    opt.cuda = False
    # Prepare DataLoader
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']  # 保存有长度数据，文件位置信息
    max_trans_len = preprocess_settings.max_word_seq_len
    test_src_word_insts = read_instances_from_file(    # 原始数据list
        opt.src,
        max_trans_len,
        preprocess_settings.keep_case)
    test_src_insts = convert_instance_to_idx_seq(      # word-->index
        test_src_word_insts, preprocess_data['dict']['src'])

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            src_insts=test_src_insts),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=collate_fn)

    translator = Translator(opt)

    with open(opt.output, 'w', encoding='utf-8') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            # batch是批量数据(单侧待翻译数据index列表) + 位置索引, *使得参数列表化
            # (列表有两部分内容，1是seq的id序列，2是对应的未知序列)
            all_hyp, all_scores = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                if not author:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seqs])  # RBX
                else:
                    idx_seqs = idx_seqs[0]
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seqs][0])
                f.write(pred_line + '\n')
    print('[Info] Finished.')

if __name__ == "__main__":
    main()

