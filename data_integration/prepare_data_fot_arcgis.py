import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from pylab import mpl

df = pd.read_csv('normalization_type_5.csv')
cityID = df['CityID'].values

# get new_type_CityID
new_type_list = df['new_type'].values
new_type_CityID = list()
new_type_set = [24, 76, 110, 131, 142]
new_type_CityID_set = [2953, 3336, 3559, 3727, 3855]

for n_t in new_type_list:
    new_type_CityID.append(new_type_CityID_set[new_type_set.index(n_t)])

# get raw_type name
raw_type = df['type'].values

df_type = pd.read_csv('citytree_type.csv')

citytree_type = df_type[df_type['CityID'].isin(cityID)]
city_name = list(citytree_type['CITYNAME'].values)
city_type = list(citytree_type['TYPE'].values)

# print(len(city_name))

for i in range(0, 13):
    city_name.append('Unknown')
    city_type.append(-1)

print(len(city_name))

period_series_df = pd.read_csv('period_series_7.csv')
period_series = period_series_df['period_series'].values
periods = period_series_df.iloc[:, 5:9].values
periods_set = sorted(list(set(periods.flatten())))
print(periods_set)

position_list = list()
for i in range(0, len(period_series)):
    position = list()
    for period in periods_set:
        position.append(period_series[i].find(str(period)))
        # count_list.append(period_series[i].count(str(period)))
    # print(position)
    position_list.append(position)

print(position_list)

# periods_set = ['fast', 'boost', 'slow', 'stable', 'stop']
periods_set = ['boost', 'fast', 'medium', 'stable', 'stop']

development_trend_list = list()
for pos in position_list:
    development_trend = list()
    for index in range(0, 13, 4):
        if index in pos:
            development_trend.append(str(periods_set[pos.index(index)]))
    development_trend_list.append(development_trend)

print(development_trend_list)
for i in range(0, len(development_trend_list)):
    development_trend_list[i] = '->'.join(development_trend_list[i])

print(development_trend_list)

df['CITY_NAME'] = city_name
df['Development_Trend'] = development_trend_list

citytree_area_df = pd.read_csv('citytree_area_1990_to_2015.csv')
citytree_area_df.insert(1, 'new_type', df['new_type'].values)
citytree_area_df.insert(1, 'new_type_CityID', new_type_CityID)
citytree_area_df.insert(1, 'type', df['type'].values)
citytree_area_df.insert(1, 'CITY_NAME', city_name)
citytree_area_df.insert(1, 'target', development_trend_list)

# periods_set = ['快速增长', '爆发式增长', '增速放缓', '平稳增长', '增长停滞']
periods_set = ['爆发式增长', '快速增长', '中速增长', '平稳增长', '低速增长']

development_trend_list = list()
for pos in position_list:
    development_trend = list()
    for index in range(0, 13, 4):
        if index in pos:
            development_trend.append(str(periods_set[pos.index(index)]))
    development_trend_list.append(development_trend)

print(development_trend_list)
for i in range(0, len(development_trend_list)):
    development_trend_list[i] = '->'.join(development_trend_list[i])

period1 = period_series_df['period1'].values
period2 = period_series_df['period2'].values
period3 = period_series_df['period3'].values
period4 = period_series_df['period4'].values

periods_list = list()
periods_list.append(period1)
periods_list.append(period2)
periods_list.append(period3)
periods_list.append(period4)

new_period1 = list()
new_period2 = list()
new_period3 = list()
new_period4 = list()

new_period_list = list()
new_period_list.append(new_period1)
new_period_list.append(new_period2)
new_period_list.append(new_period3)
new_period_list.append(new_period4)

for i in range(0, 4):
    period_num = list([367, 510, 513, 534, 672])
    for val in periods_list[i]:
        new_period_list[i].append(periods_set[period_num.index(val)])

print(new_period_list[0])
print(new_period_list[1])
print(new_period_list[2])
print(new_period_list[3])


citytree_area_df.insert(1, 'change_name', development_trend_list)
citytree_area_df['1991_to_2000'] = new_period_list[0]
citytree_area_df['1996_to_2005'] = new_period_list[1]
citytree_area_df['2001_to_2010'] = new_period_list[2]
citytree_area_df['2006_to_2015'] = new_period_list[3]

citytree_area_df.to_csv('new_trend.csv', encoding="utf_8_sig")
