# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('citytree_BA_1990_to_2015.csv')
new_df = df[(df.type > 0) & (df.type < 8) & (df.type != 5)].copy()
new_df = df[df.type == 3].copy()
type_dict = {'1': '萌芽型', '3': '成长型', '4': '发育型', '7': '区域型'}
print(len(new_df))
# df['BA_Growth'] = (df['BA2015'].values - df['BA1990'].values) / df['BA1990'].values
BA_growth = list()
for i in range(0, len(new_df)):
    if new_df['BA1990'].values[i] == 0:
        BA_growth.append(0)
    else:
        BA_growth.append((new_df['BA2015'].values[i] - new_df['BA1990'].values[i]) / new_df['BA1990'].values[i])

# BA_growth = sorted(BA_growth)

Attribute_growth_list = list()
attribute_name_list = list()
for attr in range(0, 13):
    attribute_name = list(new_df)[attr + 10][4:]
    print(attribute_name)
    attribute_name_list.append(attribute_name)
    attribute_list = list()
    for i in range(0, len(new_df)):
        if new_df['1990' + attribute_name].values[i] == 0:
            attribute_list.append(0)
        else:
            attribute_list.append(
                (new_df['2015' + attribute_name].values[i] - new_df['1990' + attribute_name].values[i]) /
                new_df['1990' + attribute_name].values[i])
    Attribute_growth_list.append(attribute_list)

print(len(Attribute_growth_list))
# Population_growth = list()
# for i in range(0, len(new_df)):
#     Population_growth.append(
#         (new_df['2015生产总值'].values[i] - new_df['1990生产总值'].values[i]) / new_df['1990生产总值'].values[i])
#
# Population_growth = sorted(Population_growth)

x = range(0, len(new_df))
plt.figure('data')

for index in range(0, 13):
    if index != 6 and index != 4:
        plt.plot(x, Attribute_growth_list[index], '-', label=attribute_name_list[index])
plt.plot(x, BA_growth, 'o', label='BA')
plt.legend(loc='best', frameon=False)
plt.xlabel('BA & attribute')
plt.ylabel('Growth')
plt.title(type_dict['1'] + ':Growth-Attribute')
plt.show()
