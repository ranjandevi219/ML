import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
iris = load_iris()
X = iris.data[:, :2]  # using only 2 features for 2D visualization
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Euclidean distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# KNN from scratch
def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    distances.sort()
    neighbors = distances[:k]
    classes = [label for _, label in neighbors]
    prediction = Counter(classes).most_common(1)[0][0]
    return prediction

# Predicting on test set
predictions = []
k = 3
for test_point in X_test:
    pred = knn_predict(X_train, y_train, test_point, k)
    predictions.append(pred)

# Accuracy
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
