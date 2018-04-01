# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('citytree_type_1990_to_2015.csv')

# df['Growth'] = (df['BA2015'] - df['BA1990']) / df['BA1990']

# df.to_csv('citytree_BA_type_1990_to_2015.csv', encoding="utf_8_sig")

new_df = df[(df.type > 0) & (df.type < 8) & (df.type != 5)].copy()
# population_list = [3, 133, 328]
# population_list = list()
# for i in range(5, len(list(new_df)) - 5, 13):
#     population_list.append(i)
# print(population_list)
# population = sum(new_df.iloc[:, population_list].values)
attr_list = list()
for i in range(3, 17):
    index_list = [i, i + 130, i + 325]
    attr = sum(new_df.iloc[:, index_list].values)
    attr_list.append(attr)
print(attr_list)
ba = sum(new_df.loc[:, ['BA1990', 'BA2000', 'BA2015']].values)
print(ba)
#
x = [1990, 2000, 2015]
plt.figure()
plt.plot(x, ba, label='BA')
# plt.plot(x, attr_list[0])
# plt.plot(x, attr_list[1])
# plt.plot(x, attr_list[2])
for i in range(0, 13):
    plt.plot(x, attr_list[i], label=i)
plt.legend(loc='upper left', frameon=False)
plt.show()
