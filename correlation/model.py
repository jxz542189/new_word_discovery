from collections import defaultdict
import numpy as np
import re
import os
import codecs
import pkuseg
from numba import jit, int32, float32
from utils.log_util import Logger
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
min_proba = {2:5, 3:25, 4:125}
write_path = os.path.join(path, 'data', 'result.txt')
min_count = 5
word_max_length = 4
prog = re.compile('[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9a-zA-Z]+')


@jit(int32(float32, int32, int32, int32), nopython=True)
def agglomeration(total, freq, freq_left, freq_right):
    return total * freq / (freq_left * freq_right)

class FindNewToken(object):
    def __init__(self, txt_path, word_max_length=word_max_length, write_path=write_path, min_count=min_count,
                 token_length=4, min_proba=min_proba):
        self.txt_path = txt_path
        self.word_max_length = word_max_length
        self.min_count = min_count
        self.token_length = token_length
        self.min_proba = min_proba
        self.write_path = write_path
        Logger.log_info.info('read text')
        self.read_text()
        Logger.log_info.info("word cut")
        self.word_cut()
        Logger.log_info.info("statistic_ngrams")
        self.statistic_ngrams()
        Logger.log_info.info("filter_ngrams")
        self.filter_ngrams()
        Logger.log_info.info("sentences_cut")
        self.sentences_cut()
        Logger.log_info.info("judge exist")
        self.judge_exist()
        Logger.log_info.info("write")
        self.write()

    def read_text(self):
        with codecs.open(self.txt_path, encoding='utf-8') as f:
            texts = f.readlines()
        texts = list(map(lambda x: x.strip(), texts))
        self.texts = list(map(lambda x: prog.sub("", x),
                              texts))

    def word_cut(self):
        seg = pkuseg.pkuseg()
        word_set = set()
        for text in self.texts:
            words = seg.cut(text)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
        self.word_set = word_set

    def statistic_ngrams(self):
        ngrams = defaultdict(int)
        for txt in self.texts:
            for char_id in range(len(txt)):
                for step in range(1, self.word_max_length + 1):
                    if char_id + step <= len(txt):
                        ngrams[txt[char_id: char_id + step]] += 1
        self.ngrams = {k: v for k, v in ngrams.items() if v >= self.min_count}
        self.total = float(sum([v for k, v in self.ngrams.items() if len(k) == 1]))



    def calculate_prob(self, token):
        if len(token) >= 2:
            score = min(
                [agglomeration(self.total, self.ngrams[token], self.ngrams[token[:i + 1]], self.ngrams[token[i + 1:]])
                 for i in range(len(token) - 1)])
            if score > self.min_proba[len(token)]:
                return True
            else:
                return False


    def filter_ngrams(self):
        self.ngrams_ = set(token for token in self.ngrams if self.calculate_prob(token))


    def cut_sentence(self, txt):
        mask = np.zeros(len(txt) - 1)
        for char_id in range(len(txt) - 1):
            for step in range(2, self.word_max_length + 1):
                if txt[char_id: char_id + step] in self.ngrams_:
                    mask[char_id: char_id+step-1] += 1
        sent_token = [txt[0]]
        for index in range(1, len(txt)):
            if mask[index-1] > 0:
                sent_token[-1] += txt[index]
            else:
                sent_token.append(txt[index])
        return txt, sent_token

    def sentences_cut(self):
        self.sentences_tokens = []
        all_tokens = defaultdict(int)
        for txt in self.texts:
            if len(txt) > 2:
                for token in self.cut_sentence(txt)[1]:
                    all_tokens[token] += 1
                self.sentences_tokens.append(self.cut_sentence(txt))
        self.all_tokens = {k: v for k, v in all_tokens.items() if v >= self.min_count}


    def is_real(self, token):
        if len(token) >= 3:
            for i in range(3, self.word_max_length + 1):
                for j in range(len(token) - i + 1):
                    if token[j:j + i] not in self.ngrams_:
                        return False
            return True
        else:
            return True


    def judge_exist(self):
        self.pairs = []
        for sent, token in self.sentences_tokens:
            real_token = []
            for tok in token:
                if self.is_real(tok) and len(tok) != 1:
                    real_token.append(tok)
            self.pairs.append((sent, real_token))

        self.new_word = {k: v for k, v in self.all_tokens.items() if self.is_real(k)}

    def statistic_token(self):
        count = defaultdict(int)
        length = list(map(lambda x: len(x), self.new_word.keys()))
        for i in length:
            count[i] += 1

    def write(self):
        with open(self.write_path, 'w', encoding='utf-8') as f:
            for sent, token in self.pairs:
                new_token = []
                for word in token:
                    if word not in self.word_set:
                        new_token.append(word)
                f.write( ','.join(new_token) + "====" + sent  +'\n')


if __name__ == '__main__':
    Logger.log_info.info("starting ...")
    txt_path = '../data/corpus.txt'
    findtoken = FindNewToken(txt_path)
    findtoken.statistic_token()
    Logger.log_info.info("ending ...")
