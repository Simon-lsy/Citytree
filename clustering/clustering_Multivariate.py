import matplotlib.pylab as plt
import numpy as np
import random
import pandas as pd


def cluster(distances, k):
    m = distances.shape[0]  # number of points

    # Pick k random medoids.
    curr_medoids = np.array([-1] * k)
    while not len(np.unique(curr_medoids)) == k:
        curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    old_medoids = np.array([-1] * k)
    new_medoids = np.array([-1] * k)

    # To be repeated until mediods stop updating
    while not ((old_medoids == curr_medoids).all()):
        # Assign each point to cluster with closest medoid.
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # Update cluster medoids to be lowest cost point.
        for curr_medoid in curr_medoids:
            cluster = np.where(clusters == curr_medoid)[0]
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)

        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        # print('Mediods still not equal')

    return clusters, curr_medoids


def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:, medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters


def compute_new_medoid(cluster, distances):
    mask = np.ones(distances.shape)
    mask[np.ix_(cluster, cluster)] = 0.
    cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    costs = cluster_distances.sum(axis=1)
    return costs.argmin(axis=0, fill_value=10e9)


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


def get_min_centroids_distance(centroids):
    centroids_distance = list()
    for i in range(0, len(centroids) - 1):
        for j in range(i + 1, len(centroids)):
            distance = 0.0
            # Find sum of distances along each dimension
            for z in range(np.shape(new_data)[2]):
                distance += DTWDistance(new_data[centroids[i]][:, z], new_data[centroids[j]][:, z], w)
            centroids_distance.append(distance)
    return centroids_distance


# a = np.array([1, 2, 1, 2, 4, 5]).tolist()
# assignments = {k: a.count(k) for k in set(a)}
# print(assignments[1])

def get_sum_distance(data_target):
    data_target = data_target.tolist()
    assignments = {k: data_target.count(k) for k in set(data_target)}
    sum_distance = 0
    for i in range(0, len(data_target)):
        distance = 0.0
        for z in range(np.shape(new_data)[2]):
            distance += DTWDistance(new_data[i][:, z], new_data[data_target[i]][:, z], w)
        # sum_distance += distance * assignments[data_target[i]]
        sum_distance += distance
    return sum_distance


df = pd.read_csv('normalization.csv')
new_df = df[df.CityID != 3536].copy()
data = new_df.iloc[:, 3:].values
print(len(data))
print(type(data))
new_data = data.reshape(199, 26, 13)

print(np.shape(new_data))

# Calculate distances using DTW
distances = np.zeros((np.shape(new_data)[0], np.shape(new_data)[0]))
# window size
w = 10

for ind, i in enumerate(new_data):
    for c_ind, j in enumerate(new_data):
        cur_dist = 0.0
        # Find sum of distances along each dimension
        for z in range(np.shape(new_data)[2]):
            cur_dist += DTWDistance(i[:, z], j[:, z], w)
        distances[ind, c_ind] = cur_dist
    print('First row completed', ind, c_ind)
print('Distances calculated')
print(distances)
sum_distance_list = list()
min_centroids_distance_list = list()
for num_cluster in range(2, 10):
    clusters, curr_medoids = cluster(distances, num_cluster)
    print('Mediods are :')
    print(curr_medoids)
    print('Cluster assigments : ')
    print(clusters)
    centroids_distance = get_min_centroids_distance(curr_medoids)
    # print(min(centroids_distance))
    min_centroids_distance_list.append(min(centroids_distance))
    sum_distance = get_sum_distance(clusters)
    # print(sum_distance)
    sum_distance_list.append(sum_distance)

print('--------------------------')
print(min_centroids_distance_list)
print(sum_distance_list)

plt.plot(range(2, 10), sum_distance_list)
plt.show()

plt.plot(range(2, 10), min_centroids_distance_list)
plt.show()

# num_cluster = 6 might be the best

# validity = list()
# for s, m in zip(sum_distance_list, min_centroids_distance_list):
#     v = m/s
#     validity.append(v)
#
# print(validity)
# plt.plot(range(2, 10), validity)
# plt.show()
