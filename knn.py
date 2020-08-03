import numpy as np

from sklearn.neighbors import KNeighborsClassifier


def knn_run(train_xs, train_ys, test_xs, k=5):
    k = min(k, len(train_ys) - 1)
    nn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    nn.fit(train_xs, train_ys)
    return nn.predict(test_xs)
