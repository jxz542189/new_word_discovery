import numba as nb
import numpy as np
from utils.log_util import Logger

def add1(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs


def add2(x, c):
    return [xx + c for xx in x]


@nb.jit(nopython=True)
def add_with_jit(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs

y = np.random.random(10 ** 5).astype(np.float32)
x = y.tolist()

Logger.log_info.info('add1')
add1(x, 1)
Logger.log_info.info('add2')
add2(x, 1)
Logger.log_info.info('add_with_jit')
add_with_jit(x, 1)
Logger.log_info.info('end')

from string import digits

s = 'abc123def456ghi789zero0'
remove_digits = str.maketrans('', '', digits)

x = "中国国家卫生健康委员会主任马晓伟20日曾对此表示，2009年至2016年"
line = str(filter(lambda x: x.isalpha(), "a1a2a3s3d4f5fg6h"))
res = x.translate(remove_digits)
print(res)
