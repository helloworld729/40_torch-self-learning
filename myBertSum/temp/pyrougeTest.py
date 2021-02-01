#!/usr/bin/env python
"""Main training workflow"""
from __future__ import division

import os, time, argparse, subprocess, shutil
import myBertSum.src.others.myPyrouge as myPyrouge

def process(temp_dir, cand, ref):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]   # 标准系统摘要
    references = [line.strip() for line in open(ref, encoding='utf-8')]  # 生成摘要
    print(len(candidates))
    print(len(references))

    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")

    for i in range(cnt):
        if len(references[i]) < 1:
            continue
        with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                  encoding="utf-8") as f:
            f.write(candidates[i])
        with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                  encoding="utf-8") as f:
            f.write(references[i])
    # 获取Rouge对象
    r = myPyrouge.Rouge155(temp_dir=temp_dir)
    # model_dir： 表示标准摘要的路径[确定，因为可以取clone的pyrouge包里面看，所以以后首先考虑来源]
    # system_dir：表示生成摘要的路径
    r.model_dir = tmp_dir + "/reference/"
    r.system_dir = tmp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = r'cand.(\d+).txt'

    # 评测
    rouge_cmd = r.convert_and_evaluate()
    ret = subprocess.call("perl " + rouge_cmd, shell=True)
    print(ret)
    # # 结果规范化
    # results_dict = r.output_to_dict(rouge_results)
    #
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)  # 递归的删除文件
    return ret

def testRouge(args, step):
    # 候选摘要地址 result文件夹
    can_path = '%s_step%d.candidate' % (args.result_path, step)
    # 最佳摘要地址 result文件夹
    gold_path = '%s_step%d.gold' % (args.result_path, step)

    result = process(args.temp_dir, can_path, gold_path)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')
    parser.add_argument("-test_from", default="../models/" + "model_step_") # 23000.pt
    parser.add_argument("-train_from", default='')
    parser.add_argument("-report_rouge",  default=True)
    parser.add_argument("-block_trigram", default=True)

    args = parser.parse_args()
    for step in ["152500", "155000", "160000", "2500"]:
        args.test_from += step + ".pt"
        testRouge(args, int(step))

"""
XML文件中，两类摘要用modle和peer表示，显然model文件是生成的
peer表示偷窥，即标准摘要

    <EVAL ID="1">
        <MODEL-ROOT>../temp\tmpo8deww0p\model</MODEL-ROOT>
        <PEER-ROOT>../temp\tmpo8deww0p\system</PEER-ROOT>
        <INPUT-FORMAT TYPE="SEE">
        </INPUT-FORMAT>
        <PEERS>
            <P ID="1">cand.0.txt</P>
        </PEERS>
        <MODELS>
            <M ID="A">ref.0.txt</M>
        </MODELS>
    </EVAL>
    
需要 smart_common_words.txt的支持
"""
