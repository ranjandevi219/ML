import numpy as np
import pandas as pd

# Load data from CSV file
# Make sure to upload your CSV file in Colab or provide the correct path
data = pd.read_csv('/content/your_data.csv')  # Replace with your CSV file path

# Select features (assuming all columns are numeric features)
X = data.values

# Euclidean distance calculation
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Initialize centroids randomly
def initialize_centroids(X, k):
    np.random.seed(42)
    random_indices = np.random.permutation(X.shape[0])
    centroids = X[random_indices[:k]]
    return centroids

# Assign clusters based on closest centroid
def assign_clusters(X, centroids):
    clusters = []
    for x in X:
        distances = [euclidean_distance(x, centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

# Update centroids based on current clusters
def update_centroids(X, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) == 0:  # Handle empty cluster
            new_centroids.append(X[np.random.choice(len(X))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

# k-Means algorithm
def k_means(X, k, max_iters=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids

# Number of clusters (choose based on your problem)
k = 3

# Run k-Means clustering
clusters, centroids = k_means(X, k)

# Add cluster labels to original data
data['Cluster'] = clusters

print("Cluster centroids:")
print(centroids)

print("\nSample clustered data:")
print(data.head())

# Save clustered data with cluster labels to CSV
data.to_csv('/content/clustered_data.csv', index=False)
