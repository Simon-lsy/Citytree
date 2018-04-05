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
        print('Mediods still not equal')

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


df = pd.read_csv('normalization.csv')
data = df.iloc[:, 3:].values
print(len(data))
print(type(data))
new_data = data.reshape(200, 26, 13)

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
clusters, curr_medoids = cluster(distances, 3)
print('Mediods are :')
print(curr_medoids)
print('Cluster assigments : ')
print(clusters)


