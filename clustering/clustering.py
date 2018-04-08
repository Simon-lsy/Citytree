import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
import math
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum = LB_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum = LB_sum + (i - lower_bound) ** 2

    return np.sqrt(LB_sum)


def k_means_clust(data, num_clust, num_iter, w=5):
    centroids = random.sample(data.tolist(), num_clust)
    counter = 0
    assignments = {}
    for n in range(num_iter):
        counter += 1
        # print(counter)
        # assign data points to clusters
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                if LB_Keogh(i, j, 5) < min_dist:
                    cur_dist = DTWDistance(i, j, w)
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust] = []

        # recalculate centroids of clusters
        for key in assignments:
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

    # print(assignments)
    return centroids, assignments


# train = np.genfromtxt('datasets/train.csv', delimiter='\t')
# test = np.genfromtxt('datasets/test.csv', delimiter='\t')
# data = np.vstack((train[:, :-1], test[:, :-1]))
#
# print(type(data))

df = pd.read_csv('new_data.csv')
# new_df = df[df.CityID != 3536].copy()
new_df = df[(df.CityID != 3536) & (df.CityID != 3489) & (df.CityID != 3326)].copy()
# print(len(new_df))
data = df.iloc[:, 2:].values
# print(data)
data_list = list()
for row in data:
    for elem in row:
        data_list.append(elem)

# print(data_list)
min_data = min(data_list)
max_data = max(data_list)
print(min_data)
print(max_data)
print(len(data_list))
sp = (sorted(data_list)[1] - min_data) / (max_data - min_data)
for i in range(0, len(data)):
    for j in range(0, 26):
        data[i][j] = (data[i][j] - min_data) / (max_data - min_data)
        if data[i][j] != 0:
            data[i][j] = math.log(data[i][j])
        else:
            data[i][j] = math.log(sp)

print(data)

# print(type(data))
for i in data:
    plt.plot(i)
plt.show()

sum_distance_list = list()
min_centroids_distance = list()
data_target_list = list()
centroids_list = list()
for num_clust in range(2, 10):
    data_target = np.zeros(len(data))
    centroids, assignments = k_means_clust(data, num_clust, 10, 4)
    print(centroids)
    centroids_list.append(centroids)
    print(assignments)
    for key in assignments:
        for index in assignments[key]:
            data_target[index] = key
    # print(data_target)
    data_target_list.append(data_target)
    sum_distance = 0
    for key in assignments:
        for index in assignments[key]:
            distance = DTWDistance(data[index], centroids[key], w=5)
            sum_distance += distance

    print(sum_distance / len(data))
    sum_distance_list.append(sum_distance / len(data))

    centroids_distance = list()
    for i in range(0, len(centroids) - 1):
        for j in range(i + 1, len(centroids)):
            distance = DTWDistance(centroids[i], centroids[j], w=5)
            centroids_distance.append(distance)

    print(min(centroids_distance))
    min_centroids_distance.append(min(centroids_distance))

    print('---------------------------')
#
#     for i in centroids:
#         plt.plot(i)
#     plt.show()

plt.plot(range(2, 10), sum_distance_list)
plt.show()

plt.plot(range(2, 10), min_centroids_distance)
plt.show()

validity = list()
for s, m in zip(sum_distance_list, min_centroids_distance):
    v = s / m
    validity.append(v)

print(validity)
plt.plot(range(2, 10), validity)
plt.show()


# data_target = np.zeros(len(data))
# centroids, assignments = k_means_clust(data, 8, 10, 4)
# for key in assignments:
#     type_list = list()
#     for index in assignments[key]:
#         # data_target[index] = key
#         type_list.append(data[index])
#     for i in type_list:
#         plt.plot(i)
#     plt.show()

min_validity_index = validity.index(min(validity))
print(centroids_list[min_validity_index])
df['type'] = data_target_list[min_validity_index]
#
df.to_csv('new_data_type.csv', encoding="utf_8_sig")

type_df = pd.read_csv('citytree_type_1990_to_2015.csv')
type_df['new_type'] = data_target_list[min_validity_index]
new_type = type_df['new_type']
type_df.drop(labels=['Unnamed: 0'], axis=1, inplace=True)
type_df.drop(labels=['Unnamed: 0.1'], axis=1, inplace=True)
type_df.drop(labels=['new_type'], axis=1, inplace=True)
type_df.insert(1, 'new_type', new_type)
type_df.to_csv('new_type.csv', encoding="utf_8_sig")