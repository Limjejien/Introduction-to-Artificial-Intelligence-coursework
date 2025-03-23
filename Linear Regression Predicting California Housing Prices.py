# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 17:26:50 2025

@author: G14
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data["MedHouseVal"] = housing.target  # Add target column

# Scatter plot of MedInc vs. MedHouseVal
plt.scatter(data["MedInc"], data["MedHouseVal"], alpha=0.1)
plt.xlabel("Median Income ($10,000s)")
plt.ylabel("Median House Value ($100,000s)")
plt.title("Housing Prices vs Median Income")
plt.show()

# Summary statistics
print(data[["MedInc", "MedHouseVal"]].describe())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data[["MedInc"]], data["MedHouseVal"], test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using Batch Gradient Descent
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Prediction for Batch Gradient Descent
y_pred = lin_reg.predict(X_test)

# Print model coefficients
print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficient: {lin_reg.coef_[0]}")

# Train using Stochastic Gradient Descent
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, learning_rate="optimal", random_state=42)
sgd_reg.fit(X_train, y_train)

# Prediction for Stochastic Gradient Descent
y_sgd_pred = sgd_reg.predict(X_test)

print(f"SGD Intercept: {sgd_reg.intercept_[0]}")
print(f"SGD Coefficient: {sgd_reg.coef_[0]}")

# Predict house value for MedInc = 8.0 ($80,000)
med_inc_example = np.array([[8.0]])  # Reshape to match input format

# Transform the input
med_inc_example_scaled = scaler.transform(med_inc_example)

# Predict using both models
predicted_value_linreg = lin_reg.predict(med_inc_example)
predicted_value_sgd = sgd_reg.predict(med_inc_example)

print(f"Predicted house value (Linear Regression) for MedInc = 8.0: ${predicted_value_linreg[0] * 100000:.2f}")
print(f"Predicted house value (SGD Regression) for MedInc = 8.0: ${predicted_value_sgd[0] * 100000:.2f}")


# Evaluate the models
mse = mean_squared_error(y_test, y_pred)
print(f"Linear Regression MSE: {mse}")

mse_sgd = mean_squared_error(y_test, y_sgd_pred)
print(f"SGD Regression MSE: {mse_sgd}")

# Plot the results
plt.scatter(X_test, y_test, color="blue", label="Actual Prices", alpha=0.2)
plt.plot(X_test, y_pred, color="red", linewidth=1, label="Predicted Prices (Batch GD)")
plt.xlabel("Median Income ($10,000s)")
plt.ylabel("Median House Value ($100,000s)")
plt.legend()
plt.title("Linear Regression: Housing Prices vs. Income")
plt.show()
