import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math

df = pd.read_csv('normalization_type.csv')

index = df.iloc[:, 0].values
# print(index)
type = df['new_type'].values
cityID = df['CityID'].values

centriod = list()

for i, t, c in zip(index, type, cityID):
    if i == t:
        centriod.append(c)

print(centriod)

citytree_type = pd.read_csv('citytree_type.csv')
new_df = citytree_type[citytree_type['CityID'].isin(centriod)]
print(len(new_df))
