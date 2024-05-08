import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from lime import lime_tabular
from sklearn.impute import KNNImputer

# Set up Matplotlib for Chinese character support
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Load the data
data = pd.read_csv("E-data_train2.csv")

# Determine which columns are numeric for imputation
numeric_columns = data.select_dtypes(include=[np.number]).columns

# Use KNNImputer to impute missing values in numeric columns
imputer = KNNImputer(n_neighbors=5)
numeric_data = imputer.fit_transform(data[numeric_columns])
data[numeric_columns] = (
    numeric_data  # Replace the numeric columns in the original dataframe
)

# Feature selection and target variable
X = data.drop(columns=["企业", "环境得分（华证）"])
y = data["环境得分（华证）"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the MLP Neural Network
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),  # Adjusted network structure
    max_iter=1000,  # Increased iteration limit
    activation="relu",  # Activation function
    learning_rate_init=0.01,  # Learning rate
    random_state=42,
)
mlp_model.fit(X_train_scaled, y_train)

# Predict
mlp_y_pred = mlp_model.predict(X_test_scaled)

# Evaluate the model
mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_y_pred))
mlp_mae = mean_absolute_error(y_test, mlp_y_pred)
mlp_r2 = r2_score(y_test, mlp_y_pred)
print("MLP Neural Network - RMSE: ", mlp_rmse)
print("MLP Neural Network - MAE: ", mlp_mae)
print("MLP Neural Network - R²: ", mlp_r2)

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, mlp_y_pred, alpha=0.75, color="red")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("MLP Neural Network - Actual vs Predicted")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
plt.show()

# Initialize LIME Explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled, feature_names=X.columns, mode="regression"
)

# Ensure using a valid index
print("Number of test samples:", len(X_test_scaled))
i = 0  # Adjust this based on available indices
exp = explainer.explain_instance(X_test_scaled[i], mlp_model.predict, num_features=10)
exp.show_in_notebook(show_table=True)
