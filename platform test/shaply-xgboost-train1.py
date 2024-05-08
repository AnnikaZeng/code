import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("E-data_train1.csv")
data.drop(columns=["Unnamed: 12"], inplace=True)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].apply(
    lambda x: x.fillna(x.mean()), axis=0
)

# 特征选择和目标变量
X = data.drop(columns=["企业", "环境得分（华证）"])
y = data["环境得分（华证）"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建并训练XGBoost模型
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 模型评估
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


# 创建SHAP解释器并计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 可视化特征重要性
shap.summary_plot(shap_values, X_test)

# 比较实际值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.75, color="red")
plt.xlabel("实际")
plt.ylabel("预测")
plt.title("实际值 vs 预测值")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
plt.show()
