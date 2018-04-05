import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA

df = pd.read_csv('normalization.csv')
new_df = DataFrame()
new_df['CityID'] = df['CityID'].values
data = df.iloc[:, 3:].values
# print(data)
print(data[0][0])

year_attr_list = list()
for index in range(0, len(data)):
    for year in range(0, len(data[index]), 13):
        year_attr = list()
        for attr in range(0, 13):
            year_attr.append(data[index][attr + year])
        year_attr_list.append(year_attr)

# print(year_attr_list)
print(len(year_attr_list))

new_year_attr_list = PCA(n_components=1).fit_transform(year_attr_list)
print(new_year_attr_list)
# print(len(new_year_attr_list))

for year in range(0, 26):
    new_year = year + 1990
    new_year_attr = list()
    for i in range(year, len(new_year_attr_list), 26):
        new_year_attr.append(new_year_attr_list[i][0])
    new_df[str(new_year)] = new_year_attr

new_df.to_csv('new_data.csv', encoding="utf_8_sig")
