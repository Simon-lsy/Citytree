import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from pandas import DataFrame
from scipy import optimize

from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def func(x, a, b):
    # return a * np.exp(-b * x) + c
    return a * x ** b
    # return a * b ** x


df = pd.read_csv('timeseries_period_new_type_7.csv')


citytree_df = pd.read_csv('citytree_type_1990_to_2015.csv')
# cityID = [3559, 3562, 3574, 3855, 3799]
cityID = [4077, 3559, 3574, 3795, 3326]

citytree_type = citytree_df[citytree_df['CityID'].isin(cityID)]
new_cityID = citytree_type['CityID'].values
print(new_cityID)
trading1991 = citytree_type['1991进出口总额'].values
trading2000 = citytree_type['2000进出口总额'].values
trading2015 = citytree_type['2015进出口总额'].values
trading_list = list()
for i in range(0, len(cityID)):
    trading_list.append([trading1991[i], trading2000[i], trading2015[i]])

print(trading_list)

x_index = [1, 10, 25]
x_range = np.arange(1, 26, 1)
trading_value = list()

for i in range(0, 5):
    # plt.plot(x_index, trading_list[i], "red")
    # # 绘制散点
    # plt.scatter(x_index, trading_list[i], s=25, alpha=0.4, marker='o')
    #
    popt, pcov = optimize.curve_fit(func, x_index, trading_list[i])
    a = popt[0]  # popt里面是拟合系数，读者可以自己help其用法
    b = popt[1]
    # c = popt[2]
    yvals = func(x_range, a, b)
    trading_value.append(list(yvals))
    # print(yvals)
    # plt.plot(x_range, yvals, "green")
    # plt.show()

year_period = [15, 10, 10, 10, 15]

for i in range(0, 5):
    new_trading_value = trading_value[i][year_period[i]:year_period[i] + 10]
    new_trading_value = new_trading_value / new_trading_value[0]
    print(list(new_trading_value))
    plt.plot(np.arange(1, 11, 1), new_trading_value, label=str(new_cityID[i]))

plt.legend()
plt.show()

# index = df['Unnamed: 0'].values
# new_type = df['new_type'].values
# cityID = df['CityID'].values
# City_Name = df['City_Name'].values
#
# centriod = list()
# city_name = list()
# centriod_index = list()
#
# for i, t, c, c_n in zip(index, new_type, cityID, City_Name):
#     if i == t:
#         centriod_index.append(i)
#         centriod.append(c)
#         if i < 200:
#             year_period = '1991--2000'
#         elif i < 400:
#             year_period = '1996--2005'
#         elif i < 600:
#             year_period = '2001--2010'
#         else:
#             year_period = '2006--2015'
#         city_name.append(year_period + c_n)
#
# print(centriod_index)
# print(centriod)
# print(city_name)
#
# centriod_num = len(centriod_index)
#
# attr_list = list(df)[6:19]
# for i in range(0, len(attr_list)):
#     attr_list[i] = attr_list[i][6:]
#
# print(len(attr_list))
# growth_list = list()
# for attr in attr_list:
#     growth = (df['year10_' + attr].values - df['year1_' + attr].values) / df['year1_' + attr].values
#     growth_list.append(growth)
#
# print(len(growth_list))
#
# new_df = DataFrame()
# new_df['CityID'] = df['CityID'].values
# new_df['City_Name'] = df['City_Name'].values
# new_df['new_type'] = df['new_type'].values
# new_df['Type'] = df['Type'].values
#
# for i in range(0, len(growth_list)):
#     new_df[attr_list[i] + 'growth'] = growth_list[i]
#
# new_df = new_df.groupby(by=['new_type']).mean()
# growth_value = new_df.iloc[:, 2:].values
#
# attr_list[4] = '投资总额'
# attr_list[5] = '零售总额'
# attr_list[12] = '可支配收入'
#
# size = 13
# x = np.arange(size)
# centriod_num = 5
#
# total_width, n = 0.9, centriod_num
# width = total_width / n
# x = x - (total_width - width) / 2
#
# for i in range(0, centriod_num):
#     plt.bar(x + i * width, growth_value[i], width=width, label=str(centriod_index[i])+':'+city_name[i])
#     for a, b in zip(x + i * width, growth_value[i]):
#         plt.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=10)
#
# plt.xlabel("Attribute")
# plt.ylabel("Growth")
# plt.legend()
# plt.xticks(range(0, 13), attr_list)
# plt.show()
