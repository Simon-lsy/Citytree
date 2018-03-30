import pandas as pd
import numpy as np
from sklearn import preprocessing
from pandas import DataFrame
import time

df_area = pd.read_csv('normalization.csv')
df_type = pd.read_csv('citytree_type.csv')

citytree_type = df_type['TYPE'].values
citytree_area_cityID = df_area['CityID'].values
print(len(citytree_type))

citytree_area_type = list()
for cityid in citytree_area_cityID:
    if cityid < len(citytree_type):
        citytree_area_type.append(citytree_type[cityid - 1])
    else:
        citytree_area_type.append(-1)

df_area['type'] = citytree_area_type
type = df_area['type']
df_area.drop(labels=['type'], axis=1, inplace=True)
df_area.insert(2, 'type', type)
df_area.to_csv('citytree_type_1990_to_2015.csv', encoding="utf_8_sig")
