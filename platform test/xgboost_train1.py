import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种常用的中文黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# Load the data
data = pd.read_csv("E-data_train1.csv")

# Remove irrelevant column
data.drop(columns=["Unnamed: 12"], inplace=True)

# Handle missing values: compute mean only for numeric columns and fill NaNs
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].apply(
    lambda x: x.fillna(x.mean()), axis=0
)

# Feature selection and target variable
X = data.drop(columns=["企业", "环境得分（华证）"])  # Adjust column name if necessary
y = data["环境得分（华证）"]  # Adjust column name as it appears in the data

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the XGBoost model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)

# Feature importances
importances = model.feature_importances_
features = X.columns
print("Feature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")

# Sort feature importances
sorted_indices = np.argsort(importances)[
    ::-1
]  # Get the indices that would sort the array
sorted_features = features[sorted_indices]
sorted_importances = importances[sorted_indices]

# Plot sorted feature importances
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Sorted Feature Importances")
plt.gca().invert_yaxis()  # Invert axis to show highest importance at the top
plt.show()

# Compare actual vs predicted values in a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.75, color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted Values")
plt.plot(
    [y.min(), y.max()], [y.min(), y.max()], "k--", lw=4
)  # Diagonal line for reference
plt.show()
