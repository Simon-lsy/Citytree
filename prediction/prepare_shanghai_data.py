import pandas as pd
import numpy as np
from pandas import DataFrame

df = pd.read_csv('citytree_area_1990_to_2015.csv')

shanghai_data = df[df.CityID == 3536].values
shanghai_data = shanghai_data[0]
shanghai_df = DataFrame()
shanghai_df['timestamp'] = range(1990, 2016)
# print(shanghai_df)
Attribute_name_list = list()
for attr in range(0, 13):
    attribute_name = list(df)[attr + 2][4:]
    Attribute_name_list.append(attribute_name)

print(Attribute_name_list)
# Attribute_name_list.append('next常住人口')

for attr_name in Attribute_name_list:
    attr_list = list()
    index = int(Attribute_name_list.index(attr_name)) + 2
    # print(int(Attribute_name_list.index(attr_name)) + 2)
    # print(len(shanghai_data))
    for i in range(index, len(shanghai_data), 13):
        attr_list.append(shanghai_data[i])
    # if index < 15:
    #
    # else:
    #     for i in range(index, len(shanghai_data), 13):
    #         attr_list.append(shanghai_data[i])
    #     attr_list.append(1799.622005)
    print(len(attr_list))
    print(attr_list)
    shanghai_df[attr_name] = attr_list
    print('----------------------')

next_population = list()
for i in range(0, 26):
    year = i + 5
    year_index = 2 + year * 13
    if year_index < len(shanghai_data):
        next_population.append(shanghai_data[year_index])
    else:
        next_population.append(1799.622005)

shanghai_df['next5常住人口'] = next_population
shanghai_df.to_csv('shanghai_data_timeseries.csv', encoding="utf_8_sig")
