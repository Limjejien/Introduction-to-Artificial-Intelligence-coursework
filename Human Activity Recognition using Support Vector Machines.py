# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 16:09:27 2025

@author: G14
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Pathing
PATH = "C:/Users/G14/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset/"
features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"

X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/y_test.txt"

# Load the feature names
features_df = pd.read_csv(features_path, sep="\\s+", header=None, names=["idx", "feature"])
feature_names = features_df["feature"].tolist()

# Make feature names unique
feature_count = defaultdict(int)
unique_feature_names = []

for feature in feature_names:
    feature_count[feature] += 1
    if feature_count[feature] > 1:
        unique_feature_names.append(f"{feature}_{feature_count[feature]}")
    else:
        unique_feature_names.append(feature)

feature_names = unique_feature_names

# Load activity labels (mapping IDs 1-6 to string names)
activity_labels_df = pd.read_csv(activity_labels_path, sep="\\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))

# Load train/test sets
X_train = pd.read_csv(X_train_path, sep="\\s+", header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, sep="\\s+", header=None, names=["Activity"])
X_test = pd.read_csv(X_test_path, sep="\\s+", header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, sep="\\s+", header=None, names=["Activity"])

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# Convert Multi-Class to Binary
def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1  # Active
    else:
        return 0  # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

# Flatten y_train for GridSearchCV
y_train_flattened = y_train["Binary"].values.ravel()

# Define the pipeline
def create_pipeline(kernel_type):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),  # reduce from 561-> 100
        ('svc', SVC(kernel=kernel_type))
    ])

# Train and evaluate SVM models with different kernels
kernels = ["linear", "poly", "rbf"]
results = {}

for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    pipeline = create_pipeline(kernel)
    pipeline.fit(X_train, y_train["Binary"])  # Train on binary labels

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test["Binary"], y_pred)
    report = classification_report(y_test["Binary"], y_pred)
    cm = confusion_matrix(y_test["Binary"], y_pred)  # Generate confusion matrix

    results[kernel] = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm  # Store confusion matrix in results
    }

    print(f"Accuracy with {kernel} kernel: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

# Compare results
print("\nComparison of SVM models with different kernels:")
for kernel, result in results.items():
    print(f"\n{kernel.capitalize()} Kernel:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print("Classification Report:")
    print(result['classification_report'])

# Define the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=50)),  # Uncomment if using PCA
    ('svc', SVC())
])

# Define the parameter grid for GridSearchCV
param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10, 100]
    },
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1],
        'svc__degree': [2, 3],
        'svc__gamma': [0.001, 0.01, 0.1]
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.001, 0.01, 0.1]
    }
]

# Perform GridSearchCV
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='accuracy',  # Use 'f1_micro' for multi-class classification
    cv=3,  # Use 3-fold cross-validation (or 5-fold if feasible)
    n_jobs=-1,  # Use all available CPU cores
    verbose=1  # Print progress
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train_flattened)

# Print the best parameters and accuracy
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Predict on the test set using the best model
y_pred_best = grid_search.predict(X_test)

# Generate confusion matrix and classification report for the best model
cm_best = confusion_matrix(y_test["Binary"], y_pred_best)
report_best = classification_report(y_test["Binary"], y_pred_best)

print("Confusion Matrix for Best Model:")
print(cm_best)
print("Classification Report for Best Model:")
print(report_best)




