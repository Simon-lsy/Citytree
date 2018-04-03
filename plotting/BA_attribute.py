# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('citytree_type_1990_to_2015.csv')

Attribute_name_list = list()
for attr in range(0, 13):
    attribute_name = list(df)[attr + 3][4:]
    Attribute_name_list.append(attribute_name)

print(Attribute_name_list)


new_df = df[(df.type > 0) & (df.type < 8) & (df.type != 5)].copy()
attr_list = list()
for i in range(3, 17):
    index_list = [i, i + 130, i + 325]
    attr = sum(new_df.iloc[:, index_list].values)
    attr_list.append(attr)
print(attr_list)
ba = sum(new_df.loc[:, ['BA1990', 'BA2000', 'BA2015']].values)


x = [1990, 2000, 2015]
plt.figure()
plt.plot(x, ba, label='BA')
for i in range(0, 13):
    plt.plot(x, attr_list[i], label=Attribute_name_list[i])
plt.legend(loc='upper left', frameon=False)
plt.xlabel('Year')
plt.ylabel('Growth')
plt.show()
