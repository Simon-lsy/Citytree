import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

# df = pd.read_csv('./raw_data/bengbu.csv')
path = 'C:/Users/lu/PycharmProjects/graduation/City/raw_data'
files = os.listdir(path)
print(files)

# for file in files:
for i in range(10,11):
    city_name = files[i].split('.')[0]
    print(city_name)
    df = pd.read_csv('./raw_data/'+files[i])
    df = df.fillna(value=0)
    X = df['GridX'].values
    Y = df['GridY'].values
    Area = df['Area'].values
    BA2015 = df['BA2015'].values
    BA2000 = df['BA2000'].values
    BA1990 = df['BA1990'].values
    BA1975 = df['BA1975'].values


    def get_close_BA(year):
        BA = df['BA' + str(year)].values
        BA25 = list()
        BA9 = list()
        Round = list()
        for i in range(0, len(df)):
            ba25 = 0
            ba9 = 0
            round_count = 0
            for x, y, land_area, ba in zip(X, Y, Area, BA):
                if x >= X[i] - 2000 and x <= X[i] + 2000 and y >= Y[i] - 2000 and y <= Y[i] + 2000:
                    ba25 += ba
                if x >= X[i] - 1000 and x <= X[i] + 1000 and y >= Y[i] - 1000 and y <= Y[i] + 1000:
                    ba9 += ba
                    if ba != 0 and (x != X[i] or y != Y[i]):
                        round_count += 1
            BA25.append(ba25)
            BA9.append(ba9)
            Round.append(round_count)
        df['BA25_' + str(year)] = BA25
        df['BA9_' + str(year)] = BA9
        df['Round' + str(year)] = Round
        df['Round' + str(year)] = Round

    get_close_BA(2015)
    get_close_BA(2000)
    get_close_BA(1990)
    get_close_BA(1975)


    def get_closest_BA25_distance(year):
        closest_buildup_distance = list()
        BA25 = df['BA25_' + year].values
        BA9 = df['BA9_' + year].values
        hasBA25 = True
        for index1 in range(0, len(df)):
            buildup_distance = list()
            if BA25[index1] / 25000000 >= 0.99:
                closest_buildup_distance.append(0)
                # closest_buildup_area.append(ba[index1]/25000000)
            else:
                for index2 in range(0, len(df)):
                    if BA25[index2] / 25000000 >= 0.99 and index1 != index2:
                        buildup_distance.append(
                            np.sqrt(pow((X[index1] - X[index2]), 2) + pow((Y[index1] - Y[index2]), 2)))
                # print(sorted(buildup_distance))
                # print(buildup_distance)
                if (len(buildup_distance)) == 0:
                    hasBA25 = False
                    closest_buildup_distance.append(0)
                else:
                    closest_buildup_distance.append(min(buildup_distance))
                # closest_buildup_area.append(sorted(buildup_distance)[0][1]/25000000)
        if hasBA25:
            df['Closest' + year + '_Distance_BA25'] = [(x - min(closest_buildup_distance)) /
                                                       (max(closest_buildup_distance) - min(closest_buildup_distance))
                                                       for x in closest_buildup_distance]
        else:
            df['Closest' + year + '_Distance_BA25'] = closest_buildup_distance


    get_closest_BA25_distance('1975')
    get_closest_BA25_distance('1990')
    get_closest_BA25_distance('2000')
    get_closest_BA25_distance('2015')


    def get_closest_BA9_distance(year):
        closest_buildup_distance = list()
        BA25 = df['BA25_' + year].values
        BA9 = df['BA9_' + year].values
        hasBA9 = True
        for index1 in range(0, len(df)):
            buildup_distance = list()
            if BA9[index1] / 9000000 >= 0.99:
                closest_buildup_distance.append(0)
                # closest_buildup_area.append(ba[index1]/25000000)
            else:
                for index2 in range(0, len(df)):
                    if BA9[index2] / 9000000 >= 0.99 and index1 != index2:
                        buildup_distance.append(
                            np.sqrt(pow((X[index1] - X[index2]), 2) + pow((Y[index1] - Y[index2]), 2)))
                # print(sorted(buildup_distance))
                # print(buildup_distance)
                if (len(buildup_distance)) == 0:
                    hasBA9 = False
                    closest_buildup_distance.append(0)
                else:
                    closest_buildup_distance.append(min(buildup_distance))
                # closest_buildup_area.append(sorted(buildup_distance)[0][1]/25000000)
        if hasBA9:
            df['Closest' + year + '_Distance_BA9'] = [(x - min(closest_buildup_distance)) /
                                                       (max(closest_buildup_distance) - min(closest_buildup_distance))
                                                       for x in closest_buildup_distance]
        else:
            df['Closest' + year + '_Distance_BA9'] = closest_buildup_distance


    get_closest_BA9_distance('1975')
    get_closest_BA9_distance('1990')
    get_closest_BA9_distance('2000')
    get_closest_BA9_distance('2015')

    df['BA2015'] = df['BA2015'] / df['Area']
    df['BA2000'] = df['BA2000'] / df['Area']
    df['BA1990'] = df['BA1990'] / df['Area']
    df['BA1975'] = df['BA1975'] / df['Area']
    df['Area'] = df['Area'] / 1000000

    df['BA25_2015'] = df['BA25_2015'] / 25000000
    df['BA25_2000'] = df['BA25_2000'] / 25000000
    df['BA25_1990'] = df['BA25_1990'] / 25000000
    df['BA25_1975'] = df['BA25_1975'] / 25000000

    df['BA9_2015'] = df['BA9_2015'] / 9000000
    df['BA9_2000'] = df['BA9_2000'] / 9000000
    df['BA9_1990'] = df['BA9_1990'] / 9000000
    df['BA9_1975'] = df['BA9_1975'] / 9000000

    df['Round2015'] = df['Round2015'] / 8
    df['Round2000'] = df['Round2000'] / 8
    df['Round1990'] = df['Round1990'] / 8
    df['Round1975'] = df['Round1975'] / 8

    df.to_csv('new_data/new_'+city_name+'.csv')