from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import os

path = 'C:/Users/lu/PycharmProjects/graduation/City/new_data'
files = os.listdir(path)
print(files)
for file in files:
    city_name = file.split('.')[0]
    print(city_name)
    df = pd.read_csv('./new_data/' + file)


    def get_Weight(year):
        if year == 2015:
            newdf = df.iloc[:, [13, 14, 15, 25, 26, 27, 28, 29, 30, 31, 32]]
        elif year == 2000:
            newdf = df.iloc[:, [16, 17, 18, 25, 26, 27, 29, 30, 31]]
        elif year == 1990:
            newdf = df.iloc[:, [19, 20, 21, 25, 26, 29, 30]]
        elif year == 1975:
            newdf = df.iloc[:, [22, 23, 24, 25, 29]]
        data_matrix = newdf.values
        kmeans = KMeans(n_clusters=10, random_state=10).fit(data_matrix)
        labels = kmeans.labels_
        centors = kmeans.cluster_centers_
        # print(centors)
        # silhouette_avg = silhouette_score(data_matrix, labels)
        # print("For n_clusters =", 10,
        #       "The average silhouette_score is :", silhouette_avg)
        weight = list()
        for i in labels:
            weight.append((centors[i][0] + centors[i][1]) / 2)
        sorted_weight = sorted(set(weight))
        new_weight = list()
        for w in weight:
            new_weight.append((sorted_weight.index(w) + 1) / 10)
        BA = df['BA' + str(year)].values
        df['new_Weight_' + str(year)] = new_weight + BA
        # print(sum(new_weight))
        # print(sum(BA))
        # print(sum(new_weight + BA))
        df['parameter' + str(year)] = df['new_Weight_' + str(year)] / sum(new_weight + BA)
        # print(max(df['parameter'+str(year)].values))
        # print(min(df['parameter' + str(year)].values))


    get_Weight(2015)
    get_Weight(2000)
    get_Weight(1990)
    get_Weight(1975)
    df.to_csv('city_weight/' + city_name + '_with_weight.csv')
