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
        if new_df['BA1990'].values[i] == 0 and new_df['BA2000'].values[i] == 0:
            BA_growth.append(0)
        elif new_df['BA1990'].values[i] == 0 and new_df['BA2000'].values[i] != 0:
            BA_growth.append((new_df['BA2015'].values[i] - new_df['BA2000'].values[i]) / new_df['BA2000'].values[i])
        else:
            BA_growth.append((new_df['BA2015'].values[i] - new_df['BA1990'].values[i]) / new_df['BA1990'].values[i])

    type_growth.append(np.mean(BA_growth))
    print(np.mean(BA_growth))

    for attr_name in Attribute_name_list:
        attribute_growth = list()
        for i in range(0, len(new_df)):
            if new_df['1990' + attr_name].values[i] == 0 and new_df['2000' + attr_name].values[i] == 0:
                attribute_growth.append(0)
            elif new_df['1990' + attr_name].values[i] == 0 and new_df['2000' + attr_name].values[i] != 0:
                attribute_growth.append(
                    (new_df['2015' + attr_name].values[i] - new_df['2000' + attr_name].values[i]) /
                    new_df['2000' + attr_name].values[i])
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

size = 14
x = np.arange(size)

total_width, n = 0.9, 4
width = total_width / n
x = x - (total_width - width) / 2

print(len(type_growth_list[1]))

for i in range(0, 4):
    type = type_list[i]
    plt.bar(x + i * width, type_growth_list[i], width=width, label=type_dict[str(type)])
    # 设置数字标签
    for a, b in zip(x + i * width, type_growth_list[i]):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

plt.xlabel('BA & attribute')
plt.ylabel('Growth')
plt.xticks(range(0, 14), Attribute_name_list)
plt.legend()
plt.title('类型增长率')


plt.show()
