# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import os

path = 'C:/Users/lu/PycharmProjects/graduation/City/city_with_attribute_data'
cities = os.listdir(path)
print(cities)

frames = list()

for city in cities:
    city_name = city.split('_')[0]
    print(city_name)
    df = pd.read_csv('./city_with_attribute_data/' + city)
    frames.append(df)

print(len(frames))
result = pd.concat(frames)
# print(len(result))
citytree_grid = result[result.CityID != 0].copy()
print(len(citytree_grid))
citytree_area = citytree_grid.groupby(by=['CityID']).sum()
citytree_area.to_csv('citytree_area.csv', encoding="utf_8_sig")
# result.to_csv('result.csv', encoding="utf_8_sig")
