# -*- coding: utf-8 -*-
"""
# @Time    : 2018/05/26 下午5:13
# @Update  : 2018/09/28 上午10:30
# @Author  : zhanzecheng/片刻
# @File    : demo.py.py
# @Software: PyCharm
"""
import os
import jieba
from information_entropy.model import TrieNode
from information_entropy.util import get_stopwords, load_dictionary, generate_ngram, save_model, load_model
from information_entropy.config import basedir

# -*- coding: utf-8 -*-
"""
# @Time    : 2018/05/26 下午5:13
# @Update  : 2018/09/28 上午10:30
# @Author  : zhanzecheng/片刻
# @File    : demo.py.py
# @Software: PyCharm
"""
import os
import pkuseg
from information_entropy.model import TrieNode
from information_entropy.util import get_stopwords, load_dictionary, generate_ngram, save_model, load_model
from information_entropy.config import basedir
from string import digits
import re
from utils.log_util import Logger

prog = re.compile("[A-Za-z0-9\!\%\[\]\,\。\☆\★\&\-\:１\.\/]")
remove_digits = str.maketrans('', '', digits)
seg = pkuseg.pkuseg()           # 以默认配置加载模型


def load_data(filename, stopwords):
    """
    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # line = line.translate(remove_digits)
            line = prog.sub("", line)
            word_list = [x for x in seg.cut(line.strip()) if x not in stopwords]
            data.append(word_list)
    return data


def load_data_2_root(data):
    Logger.log_info.info('------> 插入节点')
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
        ngrams = generate_ngram(word_list, 3)
        for d in ngrams:
            root.add(d)
    Logger.log_info.info('------> 插入成功')


if __name__ == "__main__":
    root_name = basedir + "/data/root.pkl"
    stopwords = get_stopwords()
    if os.path.exists(root_name):
        root = load_model(root_name)
    else:
        dict_name = basedir + '/data/dict.txt'
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)

    # 加载新的文章
    filename = os.path.join(basedir, 'data/corpus.txt')
    data = load_data(filename, stopwords)
    # 将新的文章插入到Root中
    load_data_2_root(data)

    # 定义取TOP5个
    topN = 50
    result, add_word, pmi_min, pmi_max = root.find_word(topN)
    # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)
    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ---->  ', score)
    print('#############################')

    # 前后效果对比
    # test_sentence = '蔡英文在昨天应民进党当局的邀请，准备和陈时中一道前往世界卫生大会，和谈有关九二共识问题'
    # print('添加前：')
    # print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))
    #
    # for word in add_word.keys():
    #     jieba.add_word(word)
    # print("添加后：")
    # print("".join([(x + '/ ') for x in jieba.cut(test_sentence, cut_all=False) if x not in stopwords]))
