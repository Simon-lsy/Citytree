import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

df = pd.read_csv('timeseries_period_new_type.csv')

index = df['Unnamed: 0'].values
new_type = df['new_type'].values
cityID = df['CityID'].values
City_Name = df['City_Name'].values

centriod = list()
city_name = list()
centriod_index = list()

for i, t, c, c_n in zip(index, new_type, cityID, City_Name):
    if i == t:
        centriod_index.append(i)
        centriod.append(c)
        if i < 200:
            year_period = '1991--2000'
        elif i < 400:
            year_period = '1996--2005'
        elif i < 600:
            year_period = '2001--2010'
        else:
            year_period = '2006--2015'
        city_name.append(year_period + c_n)

print(centriod_index)
print(centriod)
print(city_name)

centriod_num = len(centriod_index)

attr_list = list(df)[6:19]
for i in range(0, len(attr_list)):
    attr_list[i] = attr_list[i][6:]

print(len(attr_list))
growth_list = list()
for attr in attr_list:
    growth = (df['year10_' + attr].values - df['year1_' + attr].values) / df['year1_' + attr].values
    growth_list.append(growth)

print(len(growth_list[0]))

centriod_growth_list = list()
for index in centriod_index:
    centriod_growth = list()
    for i in range(0, 13):
        centriod_growth.append(growth_list[i][index])
    centriod_growth_list.append(centriod_growth)

print(len(centriod_growth_list))

# plotting

attr_list[4] = '投资总额'
attr_list[5] = '零售总额'
attr_list[12] = '可支配收入'

size = 13
x = np.arange(size)

total_width, n = 0.9, centriod_num
width = total_width / n
x = x - (total_width - width) / 2

for i in range(0, centriod_num):
    plt.bar(x + i * width, centriod_growth_list[i], width=width, label=str(centriod_index[i])+':'+city_name[i])
    for a, b in zip(x + i * width, centriod_growth_list[i]):
        plt.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=10)

plt.xlabel("Attribute")
plt.ylabel("Growth")
plt.legend()
plt.xticks(range(0, 13), attr_list)
plt.show()

# get periods data
periods = np.hsplit(new_type, 4)

new_df = pd.read_csv('citytree_type_1990_to_2015.csv')

new_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
new_df.drop(labels=['Unnamed: 0.1'], axis=1, inplace=True)
new_df.drop(labels=['Unnamed: 0.1.1'], axis=1, inplace=True)

new_df.insert(2, 'period4', periods[3])
new_df.insert(2, 'period3', periods[2])
new_df.insert(2, 'period2', periods[1])
new_df.insert(2, 'period1', periods[0])

# new_df.to_csv('timeseries_period_analysis.csv', encoding="utf_8_sig")
