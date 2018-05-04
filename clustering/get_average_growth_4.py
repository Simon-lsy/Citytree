import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('growth_type_4.csv')
new_type = sorted(set(df['new_type'].values))
print(new_type)

attr_list = list(df)[6:19]
for i in range(0, len(attr_list)):
    attr_list[i] = attr_list[i][4:]

average_attr = df.groupby(by=['new_type']).mean()
average_attr.to_csv('average_attr_4.csv', encoding="utf_8_sig")

average_growth_list = list()
for attr in attr_list:
    average_growth_list.append(average_attr[attr + 'growth'].values)

average_growth_list = np.array(average_growth_list)
print(average_growth_list[:, 0])

attr_list[4] = '投资总额'
attr_list[5] = '零售总额'
attr_list[12] = '可支配收入'

size = 13
x = np.arange(size)

centriod_num = 4

total_width, n = 0.9, centriod_num
width = total_width / n
x = x - (total_width - width) / 2

for i in range(0, centriod_num):
    plt.bar(x + i * width, average_growth_list[:, i], width=width, label=new_type[i])
    for a, b in zip(x + i * width, average_growth_list[:, i]):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)

plt.legend()
plt.xticks(range(0, 13), attr_list)
plt.show()
