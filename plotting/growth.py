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
    if new_df['BA1990'].values[i] == 0 and new_df['BA2000'].values[i] == 0:
        BA_growth.append(0)
    elif new_df['BA1990'].values[i] == 0 and new_df['BA2000'].values[i] != 0:
        BA_growth.append((new_df['BA2015'].values[i] - new_df['BA2000'].values[i]) / new_df['BA2000'].values[i])
    else:
        BA_growth.append((new_df['BA2015'].values[i] - new_df['BA1990'].values[i]) / new_df['BA1990'].values[i])

# BA_growth = sorted(BA_growth)
print(len(BA_growth))
print(BA_growth)
for i in range(0, len(BA_growth)):
    BA_growth[i] = round(BA_growth[i], 1)

Attribute_growth_list = list()
attribute_name_list = list()
for attr in range(0, 13):
    attribute_name = list(new_df)[attr + 10][4:]
    print(attribute_name)
    attribute_name_list.append(attribute_name)
    attribute_list = list()
    for i in range(0, len(new_df)):
        if new_df['1990' + attribute_name].values[i] == 0 and new_df['2000' + attribute_name].values[i] == 0:
            attribute_list.append(0)
        elif new_df['1990' + attribute_name].values[i] == 0 and new_df['2000' + attribute_name].values[i] != 0:
            attribute_list.append(
                (new_df['2015' + attribute_name].values[i] - new_df['2000' + attribute_name].values[i]) /
                new_df['2000' + attribute_name].values[i])
        else:
            attribute_list.append(
                (new_df['2015' + attribute_name].values[i] - new_df['1990' + attribute_name].values[i]) /
                new_df['1990' + attribute_name].values[i])

    Attribute_growth_list.append(attribute_list)

print(len(Attribute_growth_list))
print(Attribute_growth_list)
print(attribute_name_list)

city_list = list()
for i in range(0, len(new_df)):
    city = list()
    city.append(BA_growth[i])
    for attr in range(0, 13):
        city.append(Attribute_growth_list[attr][i])
    city_list.append(city)


# print(len(city_list))
city_list = sorted(city_list)
print(city_list)
# print([x[0] for x in city_list])
# print(sorted(city_list))
for i in range(1, 14):
    Attribute_growth_list[i-1] = [x[i] for x in city_list]

# print(Attribute_growth_list[0])

size = len(new_df)
x = np.arange(size)
plt.bar(x, Attribute_growth_list[0], label=attribute_name_list[0])
for i in range(1, 13):
    attr_bottom_list = list()
    for j in range(0, i):
        attr_bottom_list.append(Attribute_growth_list[j])
    bottom_sum_array = np.array(attr_bottom_list)
    bottom_sum = bottom_sum_array.cumsum(axis=0, dtype=None, out=None)[-1]
    plt.bar(x, Attribute_growth_list[i], bottom=bottom_sum, label=attribute_name_list[i])
plt.legend()
plt.xticks(range(0, len(new_df)), [x[0] for x in city_list])
plt.xlabel('BA Growth')
plt.ylabel('Attribute Growth')
plt.title(type_dict['1'] + ':Growth-Attribute')
plt.show()

