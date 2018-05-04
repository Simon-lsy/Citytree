import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('normalization_type_4.csv')

new_type = df['new_type'].values
new_type_list = set(new_type)
print(new_type_list)

attr_list = list(df)[5:18]
for i in range(0, len(attr_list)):
    attr_list[i] = attr_list[i][4:]

growth_1990_to_2015_list = list()
for attr in attr_list:
    growth_1990_to_2015 = (df['2015' + attr].values - df['1991' + attr].values) / df['1991' + attr].values
    growth_1990_to_2015_list.append(growth_1990_to_2015)

print(len(growth_1990_to_2015_list))

for i in range(0, len(attr_list)):
    df[attr_list[i] + 'growth'] = growth_1990_to_2015_list[i]

df.to_csv('growth_type_4.csv', encoding="utf_8_sig")
