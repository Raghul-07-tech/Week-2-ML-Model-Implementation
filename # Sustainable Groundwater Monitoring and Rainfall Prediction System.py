# Sustainable Groundwater Monitoring and Rainfall Prediction System
# Week 2 - ML Model Implementation (Linear Regression & Random Forest)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("C:\\Users\\moort\\Downloads\\state-groundwater-boards-changes-in-depth-to-water-level.csv")

# ==============================
# 2. Data Preprocessing
# ==============================
# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Extract year and month as features
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month

# Fill missing values
data.fillna(method="ffill", inplace=True)

# ==============================
# 3. Select Features & Target
# ==============================
X = data[['year', 'month', 'latitude', 'longitude', 'state_code']]
y = data['currentlevel']

# ==============================
# 4. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 5A. Linear Regression Model
# ==============================
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# Evaluation
lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)

print("\n📊 Linear Regression Results:")
print("Mean Squared Error (MSE):", lin_mse)
print("R² Score:", lin_r2)

# ==============================
# 5B. Random Forest Regressor Model
# ==============================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n🌳 Random Forest Results:")
print("Mean Squared Error (MSE):", rf_mse)
print("R² Score:", rf_r2)

# ==============================
# 6. Visualization
# ==============================
plt.figure(figsize=(14, 6))

# Linear Regression plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lin, alpha=0.5, color="blue")
plt.xlabel("Actual Groundwater Level")
plt.ylabel("Predicted Groundwater Level")
plt.title("Linear Regression - Actual vs Predicted")

# Random Forest plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="green")
plt.xlabel("Actual Groundwater Level")
plt.ylabel("Predicted Groundwater Level")
plt.title("Random Forest - Actual vs Predicted")

plt.tight_layout()
plt.show()
