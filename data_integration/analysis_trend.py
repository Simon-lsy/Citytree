import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
from pandas import DataFrame

df = pd.read_csv('new_trend.csv')

# development_trend = list(df['change_name'].values)
#
# development_type = list(set(development_trend))
#
# print(len(development_type))
# print(development_type)
#
# type_list = [1, 3, 4, 7]
#
# for type in type_list:
#     new_df = df[df.type == type].copy()
#     print(len(new_df))
#
#     trend = new_df['change_name'].values
#     development = np.zeros(12)
#     # print(development)
#     for t in trend:
#         development[development_type.index(t)] += 1
#     print(list(development))

test = df.groupby(by=['2006_to_2015']).count()
test.to_csv('test.csv', encoding="utf_8_sig")