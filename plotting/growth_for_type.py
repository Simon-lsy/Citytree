# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('citytree_BA_1990_to_2015.csv')

Attribute_name_list = list()
for attr in range(0, 13):
    attribute_name = list(df)[attr + 10][4:]
    Attribute_name_list.append(attribute_name)

print(Attribute_name_list)
# new_df = df[(df.type > 0) & (df.type < 8) & (df.type != 5)].copy()
type_list = [1, 3, 4, 7]
type_dict = {'1': '萌芽型', '3': '成长型', '4': '发育型', '7': '区域型'}
type_growth_list = list()
for type in type_list:
    new_df = df[df.type == type].copy()
    type_name = type_dict[str(type)]
    print(type_name)
    type_growth = list()
    BA_growth = list()
    for i in range(0, len(new_df)):
        if new_df['BA1990'].values[i] == 0:
            BA_growth.append(0)
        else:
            BA_growth.append((new_df['BA2015'].values[i] - new_df['BA1990'].values[i]) / new_df['BA1990'].values[i])

    type_growth.append(np.mean(BA_growth))
    print(np.mean(BA_growth))

    for attr_name in Attribute_name_list:
        attribute_growth = list()
        for i in range(0, len(new_df)):
            if new_df['1990' + attr_name].values[i] == 0:
                attribute_growth.append(0)
            else:
                attribute_growth.append(
                    (new_df['2015' + attr_name].values[i] - new_df['1990' + attr_name].values[i]) /
                    new_df['1990' + attr_name].values[i])
        attribute_growth = np.mean(attribute_growth)
        type_growth.append(np.mean(attribute_growth))
        print(attr_name + ':' + str(attribute_growth))
    type_growth_list.append(type_growth)
    print('------------------------------')

Attribute_name_list.insert(0, 'BA')
Attribute_name_list[5] = '投资总额'
Attribute_name_list[6] = '零售总额'
Attribute_name_list[13] = '可支配收入'
# print(len(type_growth_list))
for type_growth in type_growth_list:
    # print(type_growth.index)
    print(type_growth_list.index(type_growth))
    type_growth_index = type_growth_list.index(type_growth)
    type = type_list[int(type_growth_index)]
    plt.plot(range(0, 14), type_growth, 'o',
             label=type_dict[str(type)])
    # plt.annotate(type_growth[0], xy=(0, type_growth[0]), xytext=(0, type_growth[0]+500),
    #              arrowprops=dict(facecolor='black', shrink=0.05),
    #              )
    plt.legend(loc='best', frameon=False)
    plt.xlabel('BA & attribute')
    plt.ylabel('Growth')
    plt.xticks(range(0, 14), Attribute_name_list)
    plt.title(type_dict['1'] + ':Growth-Attribute')

plt.show()
