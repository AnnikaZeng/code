import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("E-data_train3+.csv")

# 使用多重插补方法处理缺失值
imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    missing_values=np.nan,
    max_iter=10,
    random_state=42,
)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# 特征选择和目标变量
X = data.drop(
    columns=[
        "企业",
        "环境得分（华证）",
        "万元营业收入综合能耗（吨标煤/万元）",
        "单位营业收入二氧化碳排放量（吨/万元）",
        "能耗变幅（%）",
    ]
)
y = data["环境得分（华证）"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义参数网格
param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2],
}

# 创建梯度提升树模型
model = GradientBoostingRegressor(random_state=42)

# 使用网格搜索进行超参数调优
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
)
grid_search.fit(X_train_scaled, y_train)

# 输出最佳参数
print("Best Parameters:", grid_search.best_params_)

# 最佳模型
best_model = grid_search.best_estimator_

# 输出特征的重要性
importances = best_model.feature_importances_
features = X.columns
print("Feature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")

# 排序特征重要性
sorted_indices = np.argsort(importances)[::-1]
sorted_features = features[sorted_indices]
sorted_importances = importances[sorted_indices]

# 绘制排序后的特征重要性
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Sorted Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# 模型预测
y_pred = best_model.predict(X_test_scaled)

# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)

# 绘制实际值与预测值的对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.75, color="red")
plt.xlabel("实际")
plt.ylabel("预测")
plt.title("实际值 vs 预测值")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
plt.show()
