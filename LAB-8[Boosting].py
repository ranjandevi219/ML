import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# For binary classification (e.g., class 0 vs others)
y = np.where(y == 0, 1, -1)  # Convert to 1 and -1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Weak Learner: Decision Stump
def decision_stump(X, y, weights):
    n_samples, n_features = X.shape
    min_error = float('inf')
    best_stump = {}

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for thresh in thresholds:
            for inequality in ['lt', 'gt']:
                preds = np.ones(n_samples)
                if inequality == 'lt':
                    preds[X[:, feature] <= thresh] = -1
                else:
                    preds[X[:, feature] > thresh] = -1

                error = np.sum(weights[y != preds])
                if error < min_error:
                    min_error = error
                    best_stump['feature'] = feature
                    best_stump['threshold'] = thresh
                    best_stump['inequality'] = inequality
                    best_stump['predictions'] = preds.copy()

    return best_stump, min_error

# AdaBoost algorithm
def adaboost(X, y, n_estimators):
    n_samples = X.shape[0]
    weights = np.full(n_samples, 1 / n_samples)
    classifiers = []

    for i in range(n_estimators):
        stump, error = decision_stump(X, y, weights)

        # Avoid division by zero
        error = max(error, 1e-10)

        alpha = 0.5 * np.log((1 - error) / error)
        stump['alpha'] = alpha
        classifiers.append(stump)

        # Update weights
        preds = stump['predictions']
        weights *= np.exp(-alpha * y * preds)
        weights /= np.sum(weights)

    return classifiers

# Make prediction with ensemble
def predict(X, classifiers):
    final_pred = np.zeros(X.shape[0])
    for stump in classifiers:
        feature = stump['feature']
        thresh = stump['threshold']
        inequality = stump['inequality']
        alpha = stump['alpha']

        preds = np.ones(X.shape[0])
        if inequality == 'lt':
            preds[X[:, feature] <= thresh] = -1
        else:
            preds[X[:, feature] > thresh] = -1

        final_pred += alpha * preds
    return np.sign(final_pred)

# Train model
classifiers = adaboost(X_train, y_train, n_estimators=10)

# Predict and evaluate
y_pred = predict(X_test, classifiers)
accuracy = np.mean(y_pred == y_test) * 100
print(f"Boosting Accuracy: {accuracy:.2f}%")
