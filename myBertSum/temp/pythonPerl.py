import subprocess
# ret = subprocess.call(["perl", "I:/Anaconda/pyoruge/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl"])  # 第4行的意义
# ret = subprocess.call(["perl", "J:/40_torch-self-learning/baseNlp/test.pl"])  # 测试文件
cmdLine = 'perl I:/Anaconda/pyoruge/pyrouge/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl -e I:/Anaconda/pyoruge/pyrouge/tools/ROUGE-1.5.5/data -c 95 -m -r 1000 -n 2 -a ../temp/tmp96gyring/rouge_conf.xml'

ret = subprocess.call(cmdLine, shell=True)
print(ret)





# [-a(evaluate all systems)]
# [-e ROUGE_EVAL_HOME]
# [-m(use Porter stemmer)]
# [-n max - ngram]
# [-r number - of - samples(
