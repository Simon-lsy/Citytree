import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('timeseries_period_analysis.csv')

df['period_series'] = df['period1'].map(str) + '-' + df['period2'].map(str) + '-' + df['period3'].map(str) + '-' + \
                      df['period4'].map(str)

period_series = df['period_series'].values

df.drop(labels=['period_series'], axis=1, inplace=True)
df.insert(2, 'period_series', period_series)

df.to_csv('period_series.csv', encoding="utf_8_sig")
# type_list = [1, 3, 4, 7]
#
# periods = df.iloc[:, 3:7].values
# periods_set = sorted(list(set(periods.flatten())))
# print(periods_set)
#
# frequency_df = DataFrame()
# frequency_df['centriod'] = periods_set
#
# for type in type_list:
#     new_df = df[df.type == type].copy()
#     periods = new_df.iloc[:, 3:7].values
#     periods_list = list()
#     for i in range(0, len(periods)):
#         for ele in periods[i]:
#             periods_list.append(ele)
#
#     frequency = list()
#     for period in periods_set:
#         # print(period)
#         print(periods_list.count(period) / len(periods_list))
#         frequency.append(periods_list.count(period) / len(periods_list))
#     print('--------------------------')
#     frequency_df['type' + str(type)] = frequency
#
# frequency_df.to_csv('frequency_of_occurrence.csv', encoding="utf_8_sig")
