import numpy as np
import pandas as pd

# Load your dataset (replace path with your CSV file)
data = pd.read_csv('/content/your_data.csv')  # Update path accordingly

# Extract features as numpy array
X = data.values

# Step 1: Standardize (mean = 0)
X_meaned = X - np.mean(X, axis=0)

# Step 2: Compute covariance matrix
cov_matrix = np.cov(X_meaned, rowvar=False)

# Step 3: Eigen decomposition
eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)  # eigh for symmetric matrices

# Step 4: Sort eigenvectors by eigenvalues descending
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:, sorted_index]

# Select number of components (k)
k = 2  # choose k as needed
eigenvector_subset = sorted_eigenvectors[:, 0:k]

# Step 5: Project data
X_reduced = np.dot(X_meaned, eigenvector_subset)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)

# Optional: create a DataFrame for the reduced data
reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(k)])
print(reduced_df.head())
