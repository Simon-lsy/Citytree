# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
from scipy import optimize
import os


# 直线方程函数
def f_1(x, A, B):
    return A * x + B


df = pd.read_csv('citytree_area.csv')
new_df = DataFrame()
new_df['CityID'] = df['CityID'].values
attribute_name = list(df)[1:]
attributes = list()
# print(attribute_name)
for attribute in attribute_name:
    attributes.append(attribute[4:])

attributes = attributes[0:13]
print(len(attributes))
print(attributes)

for year in range(0, 10):
    for attr in range(1, 14):
        attribute = attributes[attr - 1]
        print(attribute)
        year_attr_data = list()
        for city in range(0, len(df)):
            x0 = [0, 10]
            y0 = [df.iloc[city, [attr, attr + 13]][0], df.iloc[city, [attr, attr + 13]][1]]
            A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
            x1 = np.arange(0, 10, 1)
            y1 = A1 * x1 + B1
            year_attr_data.append(y1[year])
        new_df['199' + str(year) + attribute] = year_attr_data



for year in range(0, 16):
    for attr in range(1, 14):
        attribute = attributes[attr - 1]
        print(attribute)
        year_attr_data = list()
        for city in range(0, len(df)):
            x0 = [0, 15]
            y0 = [df.iloc[city, [attr + 13, attr + 26]][0], df.iloc[city, [attr + 13, attr + 26]][1]]
            A1, B1 = optimize.curve_fit(f_1, x0, y0)[0]
            x1 = np.arange(0, 16, 1)
            y1 = A1 * x1 + B1
            year_attr_data.append(y1[year])
        if year >= 10:
            new_df['20' + str(year) + attribute] = year_attr_data
        else:
            new_df['200' + str(year) + attribute] = year_attr_data

new_df.to_csv('citytree_area_1990_to_2015.csv', encoding="utf_8_sig")
