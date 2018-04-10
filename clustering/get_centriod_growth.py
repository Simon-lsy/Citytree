import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('centriod_data_6.csv')
index = df['CityID'].values
year_attr = list(df)[3:16]
attr_list = list()
for attr in year_attr:
    attr_list.append(attr[4:])

print(attr_list)

growth_1990_to_2015_list = list()
for attr in attr_list:
    growth_1990_to_2000 = (df['2000' + attr].values - df['1990' + attr].values) / df['1990' + attr].values
    growth_2000_to_2015 = (df['2015' + attr].values - df['2000' + attr].values) / df['2000' + attr].values
    growth_1990_to_2015 = (df['2015' + attr].values - df['1990' + attr].values) / df['1990' + attr].values
    # print(growth_1990_to_2000)
    # print(growth_2000_to_2015)
    growth_1990_to_2015_list.append(growth_1990_to_2015)
    # print('-------------------')

attr_list[4] = '投资总额'
attr_list[5] = '零售总额'
attr_list[12] = '可支配收入'

size = 13
x = np.arange(size)

centriod_num = 6

total_width, n = 0.9, centriod_num
width = total_width / n
x = x - (total_width - width) / 2

for i in range(0, centriod_num):
    plt.bar(x + i * width, [x[i] for x in growth_1990_to_2015_list], width=width, label=index[i])
    for a, b in zip(x + i * width, [x[i] for x in growth_1990_to_2015_list]):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

plt.legend()
plt.xticks(range(0, 13), attr_list)
plt.show()
