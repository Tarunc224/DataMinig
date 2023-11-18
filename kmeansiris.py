import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
data = pd.read_csv("iris.csv")  # Replace "your_dataset.csv" with the actual file path

# Select the features for clustering
X = data.iloc[:, :-1].values

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow method graph to find the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster-Sum-of-Squares
plt.show()

# Based on the Elbow method, choose the optimal number of clusters (e.g., 3)
optimal_num_clusters = 3

# Apply K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_clusters = kmeans.fit_predict(X)

# Add the cluster labels to the DataFrame
data['Cluster'] = pred_clusters

# Print the results
print(data)

# Visualize the clusters (for 2D data)
plt.scatter(X[pred_clusters == 0, 0], X[pred_clusters == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[pred_clusters == 1, 0], X[pred_clusters == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[pred_clusters == 2, 0], X[pred_clusters == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()