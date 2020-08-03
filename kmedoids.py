# https://github.com/SachinKalsi/kmedoids
from scipy.sparse import csr_matrix
import numpy as np


class KMedoids:
    def __init__(self, n_cluster=2, max_iter=10, tol=0.1):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tol = tol

        self.medoids = []
        self.clusters = {}
        self.tol_reached = float('inf')
        self.current_distance = 0

        self.__data = None
        self.__rows = 0
        self.cluster_distances = {}

    def fit(self, data):
        self.__data = data
        self.__rows = len(data)
        self.__start_algo()
        return self

    def __start_algo(self):
        self.__initialize_medoids()
        self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
        self.__update_clusters()

    def __update_clusters(self):
        for i in range(self.max_iter):
            print(f'iter {i + 1}/{self.max_iter}')
            cluster_dist_with_new_medoids = self.__swap_and_recalculate_clusters()
            if self.__is_new_cluster_dist_small(cluster_dist_with_new_medoids) == True:
                self.clusters, self.cluster_distances = self.__calculate_clusters(self.medoids)
            else:
                break

    def __is_new_cluster_dist_small(self, cluster_dist_with_new_medoids):
        existance_dist = self.calculate_distance_of_clusters()
        new_dist = self.calculate_distance_of_clusters(cluster_dist_with_new_medoids)

        if existance_dist > new_dist and (existance_dist - new_dist) > self.tol:
            self.medoids = cluster_dist_with_new_medoids.keys()
            return True
        return False

    def calculate_distance_of_clusters(self, cluster_dist=None):
        if cluster_dist == None:
            cluster_dist = self.cluster_distances
        dist = 0
        for medoid in cluster_dist.keys():
            dist += cluster_dist[medoid]
        return dist

    def __swap_and_recalculate_clusters(self):
        # http://www.math.le.ac.uk/people/ag153/homepage/KmeansKmedoids/Kmeans_Kmedoids.html
        cluster_dist = {}
        for medoid in self.medoids:
            is_shortest_medoid_found = False
            for data_index in self.clusters[medoid]:
                if data_index != medoid:
                    cluster_list = list(self.clusters[medoid])
                    cluster_list[self.clusters[medoid].index(data_index)] = medoid
                    new_distance = self.calculate_inter_cluster_distance(data_index, cluster_list)
                    if new_distance < self.cluster_distances[medoid]:
                        cluster_dist[data_index] = new_distance
                        is_shortest_medoid_found = True
                        break
            if is_shortest_medoid_found == False:
                cluster_dist[medoid] = self.cluster_distances[medoid]
        return cluster_dist

    def calculate_inter_cluster_distance(self, medoid, cluster_list):
        dists = self.__data[cluster_list][:, None, :] - self.__data[[medoid]][None, :]
        dists = np.sqrt((dists ** 2).sum(axis=2)) # (#cluster_list * 1)
        return dists.mean()

    def __calculate_clusters(self, medoids):
        clusters = {}
        cluster_distances = {}
        for medoid in medoids:
            clusters[medoid] = []
            cluster_distances[medoid] = 0

        medoids = list(medoids)
        dists = self.__data[:, None, :] - self.__data[medoids][None, :]
        dists = np.sqrt((dists ** 2).sum(axis=2)) # (row * #medoids)
        nearest_medoids = dists.argmin(axis=1)
        nearest_distances = dists.min(axis=1)
        for row in range(self.__rows):
            _medoid = medoids[nearest_medoids[row]]
            cluster_distances[_medoid] += nearest_distances[row]
            clusters[_medoid].append(row)

        for medoid in medoids:
            cluster_distances[medoid] /= len(clusters[medoid])
        return clusters, cluster_distances

    def __initialize_medoids(self):
        self.medoids.append(0)
        while len(self.medoids) != self.n_cluster:
            dists = self.__data[:, None, :] - self.__data[self.medoids][None, :]
            dists = np.sqrt((dists ** 2).sum(axis=2)).min(axis=1)
            self.medoids.append(np.argsort(dists)[int(self.__rows * .8)])
