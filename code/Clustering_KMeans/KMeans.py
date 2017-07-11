# generated 2D classification datasets
from sklearn.datasets import make_blobs

# This module for data visualization
import matplotlib.pyplot as plt

# This module to import the KMeans
from sklearn.cluster import KMeans

# use the make_blobs randomly generated 2D clustering datasets.
# X is the instances and y are the labels of these instances
X, y = make_blobs(n_samples=150,  n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0 )

# plot the data using the plot module form matplotlib

plt.scatter(X[:,0], X[:,1], c='white', marker='o', s=50)
plt.grid()
plt.show()


# initialize the KMeans clustering algorithm
# number of clusters is 3
# use random method to select the cluster centers, another option is to set init='k-means++'
# run the k-means clustering algorithms 10 times independently with different random
# centroids to choose the final model as the one with the lowest SSE
# set the max number of iterations to 300
# set the convergence threshold to le-04, which is 0.0001

km = KMeans(n_clusters= 3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)

# execute the KMeans
y_km = km.fit_predict(X)

# plot the output clusters
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s=50, c='lightgreen', marker='s', label='cluster 1')

plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s=50, c='orange', marker='o', label='cluster 2')

plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s=50, c='lightblue', marker='v', label='cluster 3')


# use the following to plot additional  clusters if required
# plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s=50, c='lightyellow', marker='d', label='cluster 4')

# plt.scatter(X[y_km == 4, 0], X[y_km == 4, 1], s=50, c='brown', marker='+', label='cluster 5')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=250, marker='*', c='red', label='centroids')

plt.legend()
plt.grid()
plt.show()