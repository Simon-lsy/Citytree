import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd
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
data = df.iloc[:, 3:].values
print(type(data))
# for i in data:
#     plt.plot(i)
# plt.show()

sum_distance_list = list()
min_centroids_distance = list()
for num_clust in range(3, 10):
    data_target = np.zeros(len(data))
    # print(data_target)
    centroids, assignments = k_means_clust(data, num_clust, 10, 4)
    # print(centroids)
    # print(assignments)
    for key in assignments:
        # print(key)
        for index in assignments[key]:
            data_target[index] = key
    # print(data_target)
    sum_distance = 0
    for key in assignments:
        for index in assignments[key]:
            distance = DTWDistance(data[index], centroids[key], w=5)
            sum_distance += distance * len(assignments[key])

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

plt.plot(range(3, 10), sum_distance_list)

plt.show()

plt.plot(range(3, 10), min_centroids_distance)
plt.show()

# data_target = np.zeros(len(data))
# centroids, assignments = k_means_clust(data, 4, 10, 4)
# for key in assignments:
#     # print(key)
#     for index in assignments[key]:
#         data_target[index] = key

# print(centroids[0])
# print(len(centroids[0]))


# df['type'] = data_target
#
# df.to_csv('new_data_type.csv', encoding="utf_8_sig")
