import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

scalar_values = np.linspace(0, 1, 10)
colors = cm.viridis(scalar_values)

def display_2d(X, y):
    _, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y)
    _ = ax.legend(
        scatter.legend_elements()[0], np.unique(y), loc="lower right", title="Classes"
    )

def display_2d_clusters(X_clustered, cluster_centers, cluster_labels):
    n_clusters = len(cluster_labels)
    _, ax = plt.subplots()
    #scatter = ax.scatter(X[:, 0], X[:, 1])

    for _ in range(n_clusters):
        X_group = X_clustered[_]
        for Xs in X_group:    
            ax.scatter(Xs[:, 0], Xs[:, 1], color=colors[cluster_labels[_]])
        ax.scatter(cluster_centers[_, 0], cluster_centers[_, 1], color='r', marker='x')

