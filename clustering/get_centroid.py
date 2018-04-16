import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math


df = pd.read_csv('normalization_type_5.csv')

centriod_num = 5

index = df.iloc[:, 0].values
# print(index)
type = df['new_type'].values
cityID = df['CityID'].values

centriod = list()

for i, t, c in zip(index, type, cityID):
    if i == t:
        print(i)
        centriod.append(c)
        # print(df.iloc[i, :].values)

print(centriod)

citytree_type = pd.read_csv('citytree_type.csv')
new_df = citytree_type[citytree_type['CityID'].isin(centriod)]
# print(len(new_df))
# print(new_df.iloc[:, 0:13])
centriod_df = new_df.iloc[:, 0:13]
# print(centriod)
city_name = centriod_df['CITYNAME'].values
print(city_name)
first_area = centriod_df['FIRIST_AREA'].values
growth_year = centriod_df['GROWTH_Year'].values

size = centriod_num
x = np.arange(size)
print(x)
# x = range(0,size)

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

for i in range(0, len(growth_year)):
    growth_year[i] = float(growth_year[i][:-1])
print(growth_year)

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.bar(x, first_area, width=width, label='first area', fc='r')
for a, b in zip(x, first_area):
    plt.text(a, b + 0.05, '%.1f' % b, ha='center', va='bottom', fontsize=10)
ax1.legend(loc=2)
ax1.set_ylabel('First Area')
ax2 = ax1.twinx()  # this is the important function
plt.bar(x + width, growth_year, width=width, label='growth/year', fc='b')
for a, b in zip(x + width, growth_year):
    plt.text(a, b + 0.05, '%.1f%%' % b, ha='center', va='bottom', fontsize=10)
ax2.legend(loc=2)
ax2.set_ylabel('Growth/Year')
ax2.set_xlabel('City')
plt.legend()
plt.xticks(range(0, centriod_num), city_name)
plt.show()

citytree_area_df = pd.read_csv('citytree_area_1990_to_2015.csv')
centriod_df = citytree_area_df[citytree_area_df['CityID'].isin(centriod)]
centriod_df['CITYNAME'] = city_name
print('-------------')
print(centriod_df)
centriod_df.to_csv('centriod_data_' + str(centriod_num) + '.csv', encoding="utf_8_sig")
