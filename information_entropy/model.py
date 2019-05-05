# -*- coding: utf-8 -*-
"""
# @Time    : 2018/5/26 下午5:03
# @Author  : zhanzecheng
# @File    : model.py
# @Software: PyCharm
"""
import math
from numba import jit, int32, float32,float64
from utils.log_util import Logger
import numpy as np
#(values[0] + min(left[d], right[d])) * values[1]

@jit(float32(float32, float32, float32, float32), nopython=True)
def get_score(value_0, left_value, right_value, value_1):
    return (value_0 + min(left_value, right_value)) * value_1

#(ch.count / total) * math.log(ch.count / total, 2)
@jit(float32(float32), nopython=True)
def get_entropy(p):
    res = float32(p * np.log(p))
    return res


class Node(object):
    """
    建立字典树的节点
    """

    def __init__(self, char):
        self.char = char
        # 记录是否完成
        self.word_finish = False
        # 用来计数
        self.count = 0
        # 用来存放节点
        self.child = {}
        # 方便计算 左右熵
        # 判断是否是后缀（标识后缀用的，也就是记录 b->c->a 变换后的标记）
        self.isback = False


class TrieNode(object):
    """
    建立前缀树，并且包含统计词频，计算左右熵，计算互信息的方法
    """

    def __init__(self, node, data=None, PMI_limit=20):
        """
        初始函数，data为外部词频数据集
        :param node:
        :param data:
        """
        self.root = Node(node)
        self.PMI_limit = PMI_limit
        if not data:
            return
        node = self.root
        for key, values in data.items():
            new_node = Node(key)
            new_node.count = int(values)
            new_node.word_finish = True
            if key not in node.child.keys():
                node.child[key] = new_node

    def add(self, word):
        """
        添加节点，对于左熵计算时，这里采用了一个trick，用a->b<-c 来表示 cba
        具体实现是利用 self.isback 来进行判断
        :param word:
        :return:  相当于对 [a, b, c] a->b->c, [b, c, a] b->c->a
        """
        node = self.root
        # 正常加载
        for count, char in enumerate(word):
            if char in node.child.keys():
                node = node.child[char]
            else:
                new_node = Node(char)
                node.child[char] = new_node
                node = new_node

            # 判断是否是最后一个节点，这个词每出现一次就+1
            if count == len(word) - 1:
                node.count += 1
                node.word_finish = True

        # 建立后缀表示
        length = len(word)
        node = self.root
        if length == 3:
            word = list(word)
            word[0], word[1], word[2] = word[1], word[2], word[0]

            for count, char in enumerate(word):
                found_in_child = False
                # 在节点中找字符（不是最后的后缀词）
                if count != length - 1:
                    if char in node.child.keys():
                        node = node.child[char]
                        found_in_child = True
                else:
                    # 由于初始化的 isback 都是 False， 所以在追加 word[2] 后缀肯定找不到
                    if char in node.child.keys() and node.child[char].isback:
                        node = node.child[char]
                        found_in_child = True

                # 顺序在节点后面添加节点。 b->c->a
                if not found_in_child:
                    new_node = Node(char)
                    node.child[char] = new_node
                    node = new_node

                # 判断是否是最后一个节点，这个词每出现一次就+1
                if count == len(word) - 1:
                    node.count += 1
                    node.isback = True
                    node.word_finish = True

    def search_one(self):
        """
        计算互信息: 寻找一阶共现，并返回词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        # 计算 1 gram 总的出现次数
        total = 0
        for key in node.child.keys():
            if node.child[key].word_finish == True:
                total = total + node.child[key].count

        # 计算 当前词 占整体的比例
        for key in node.child.keys():
            if node.child[key].word_finish == True:
                result[node.child[key].char] = node.child[key].count / total
        return result, total

    def search_bi(self):
        """
        计算互信息: 寻找二阶共现，并返回 log2( P(X,Y) / (P(X) * P(Y)) 和词概率
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        total = 0
        # 1 grem 各词的占比，和 1 grem 的总次数
        one_dict, total_one = self.search_one()
        for key in node.child.keys():
            child = node.child[key]
            for k in child.child.keys():
                ch = child.child[k]
                if ch.word_finish == True:
                    total += ch.count

        PMI_min = 1000
        PMI_max = 0
        for key in node.child.keys():
            child = node.child[key]
            for k in child.child.keys():
                ch = child.child[k]
                PMI = math.log(max(ch.count, 1), 2) - math.log(total, 2) - math.log(one_dict[child.char],
                                                                                    2) - math.log(
                    one_dict[ch.char], 2)
                # 这里做了PMI阈值约束
                if PMI > PMI_max:
                    PMI_max = PMI
                if PMI < PMI_min:
                    PMI_min = PMI
                if PMI > self.PMI_limit:
                    result[child.char + '_' + ch.char] = (PMI, ch.count / total)
        return result, PMI_min, PMI_max

    def search_left(self):
        """
        寻找左频次
        统计左熵， 并返回左熵 (bc - a 这个算的是 abc|bc 所以是左熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for key in node.child.keys():
            child = node.child[key]
            for k in child.child.keys():
                cha = child.child[k]
                total = 0
                p = 0.0
                for k1 in cha.child.keys():
                    ch = cha.child[k1]
                    if ch.word_finish == True and ch.isback:
                        total += ch.count
                for k1 in cha.child.keys():
                    ch = cha.child[k1]
                    if ch.word_finish == True and ch.isback:
                        percent = ch.count / total
                        p += get_entropy(percent)
                # 计算的是信息熵
                result[child.char + cha.char] = -p
        return result

    def search_right(self):
        """
        寻找右频次
        统计右熵，并返回右熵 (ab - c 这个算的是 abc|ab 所以是右熵)
        :return:
        """
        result = {}
        node = self.root
        if not node.child:
            return False, 0

        for key in node.child.keys():
            child = node.child[key]
            for k in child.child.keys():
                cha = child.child[k]
                total = 0
                p = 0.0
                for k1 in cha.child.keys():
                    ch = cha.child[k1]
                    if ch.word_finish == True and not ch.isback:
                        total += ch.count
                for k1 in cha.child.keys():
                    ch = cha.child[k1]
                    if ch.word_finish == True and not ch.isback:
                        percent = ch.count / total
                        p += get_entropy(percent)
                result[child.char + cha.char] = -p
        return result

    def find_word(self, N):
        # 通过搜索得到互信息
        # 例如: dict{ "a_b": (PMI, 出现概率), .. }
        Logger.log_info.info('search_bi')
        bi, pmi_min, pmi_max = self.search_bi()
        # 通过搜索得到左右熵
        Logger.log_info.info('search left')
        left = self.search_left()
        Logger.log_info.info('search_right')
        right = self.search_right()
        Logger.log_info.info('result dict achieve')
        result = {}
        for key, values in bi.items():
            d = "".join(key.split('_'))
            # 计算公式 score = PMI + min(左熵， 右熵) => 熵越小，说明越有序，这词再一次可能性更大！
            result[key] = get_score(values[0], left[d], right[d], values[1])
        # 按照 大到小倒序排列，value 值越大，说明是组合词的概率越大
        # result变成 => [('世界卫生_大会', 0.4380419441616299), ('蔡_英文', 0.28882968751888893) ..]
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print("result: ", result)
        dict_list = [result[0][0]]
        # print("dict_list: ", dict_list)
        add_word = {}
        new_word = "".join(dict_list[0].split('_'))
        # 获得概率
        add_word[new_word] = result[0][1]

        Logger.log_info.info("top5")
        # 取前5个
        # [('蔡_英文', 0.28882968751888893), ('民进党_当局', 0.2247420989996931), ('陈时_中', 0.15996145099751344), ('九二_共识', 0.14723726297223602)]
        for d in result[1: N]:
            flag = True
            for tmp in dict_list:
                pre = tmp.split('_')[0]
                # 新出现单词后缀，再老词的前缀中 or 如果发现新词，出现在列表中; 则跳出循环
                # 前面的逻辑是： 如果A和B组合，那么B和C就不能组合(这个逻辑有点问题)，例如：`蔡_英文` 出现，那么 `英文_也` 这个不是新词
                # 疑惑: **后面的逻辑，这个是完全可能出现，毕竟没有重复**
                if d[0].split('_')[-1] == pre or "".join(tmp.split('_')) in "".join(d[0].split('_')):
                    flag = False
                    break
            if flag:
                new_word = "".join(d[0].split('_'))
                add_word[new_word] = d[1]
                dict_list.append(d[0])

        return result, add_word, pmi_min, pmi_max
