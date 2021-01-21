import jieba
sen = '东南大学简称“东大”，是中华人民共和国教育部直属、中央直管副部级建制的全国重点大学'
seg_list = jieba.cut(sen, cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut(sen, cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut_for_search(sen)
print("Search Mode: " + "/ ".join(seg_list))  # 搜索引擎模式