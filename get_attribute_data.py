# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

path_city = 'C:/Users/lu/PycharmProjects/graduation/City/attribute_city'
cities = os.listdir(path_city)
print(cities)

path_attributes = 'C:/Users/lu/PycharmProjects/graduation/City/attribute_data'
attributes = os.listdir(path_attributes)
print(attributes)

for city, attribute in zip(cities, attributes):
    # print(city)
    # print(attribute)
    city_name = attribute.split('.')[0]
    print(city_name)
    print(attribute)
    df_city = pd.read_csv('./attribute_city/' + city)
    df_attribute = pd.read_csv('./attribute_data/' + attribute)
    attribute_name = list(df_attribute)[1:]
    print(attribute_name)


    def get_year_data(year):
        parameter = df_city['parameter' + str(year)].values
        # print(len(parameter))
        # print(parameter)
        if year == 1990:
            index = 0
        elif year == 2000:
            index = 1
        elif year == 2015:
            index = 2
        attribute_data = df_attribute.iloc[index,][1:]
        # print(parameter)
        # print(len(attribute_data))
        # print(attribute_data)
        # print('------------------------')
        for i in range(0, len(attribute_data)):
            # print(attribute_data)
            # print(len(attribute_data))
            # print(attribute_data[i])
            # print('------------------------')
            df_city[str(year) + attribute_name[i]] = parameter * attribute_data[i]
            print(len(df_city[str(year) + attribute_name[i]].values))

    get_year_data(1990)
    get_year_data(2000)
    get_year_data(2015)

    df_city.to_csv('city_with_attribute_data/' + city_name + '_attribute_data.csv', encoding="utf_8_sig")
