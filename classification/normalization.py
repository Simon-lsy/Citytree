import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time

df = pd.read_csv('citytree_area_1990_to_2015.csv')

for attr in range(2, 15):
    index = attr
    attr_list = []
    while index < len(list(df)):
        # print(index)
        attr_list = np.append(attr_list, df.iloc[:, index].values)
        index += 13
    print('--------------------')
    max_attr = max(attr_list)
    min_attr = min(attr_list)
    # if min_attr < 0:
    #     min_attr = 0
    print(max_attr)
    print(min_attr)
    index = attr
    while index < len(list(df)):
        attribute = list(df)[index]
        print(attribute)
        df[attribute] = [(x - min_attr) /
                         (max_attr - min_attr)
                         for x in df.iloc[:, index].values]
        index += 13

df.to_csv('normalization.csv', encoding="utf_8_sig")
