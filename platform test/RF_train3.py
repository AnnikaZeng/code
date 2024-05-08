import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("E-data_train3.csv")

# 在插补过程中排除 '企业' 列
data_to_impute = data.drop(columns=["企业"])

# 识别非数值列并使用标签编码进行转换
label_encoders = {}
for column in data_to_impute.columns:
    if data_to_impute[column].dtype == object:
        le = LabelEncoder()
        data_to_impute[column] = le.fit_transform(data_to_impute[column].astype(str))
        label_encoders[column] = le

# 初始化 KNN 插补器
knn_imputer = KNNImputer(n_neighbors=5)

# 应用 KNN 插补
knn_imputed_data = knn_imputer.fit_transform(data_to_impute)

# 将插补后的数据转换回 DataFrame
knn_imputed_data = pd.DataFrame(knn_imputed_data, columns=data_to_impute.columns)

# 启用 IterativeImputer
iterative_imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    missing_values=np.nan,
    max_iter=10,
    random_state=42,
)

# 应用多重插补
full_imputed_data = iterative_imputer.fit_transform(knn_imputed_data)

# 将完全插补后的数据转换回 DataFrame
full_imputed_data = pd.DataFrame(full_imputed_data, columns=data_to_impute.columns)

# 特征选择和目标变量
X = full_imputed_data.drop(columns=["环境得分（华证）"])
y = full_imputed_data["环境得分（华证）"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 参数网格
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "oob_score": [True],
}

# 网格搜索
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
)
grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best MSE: {-grid_search.best_score_:.2f}")

# 使用最佳参数训练模型
rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
rf.fit(X_train_scaled, y_train)

# 获取特征重要性并排序
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]


# 打印特征重要性
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]:.3f})")

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.show()


# 不同特征子集的性能评估
mse_scores, mae_scores, r2_scores = [], [], []
for k in range(1, len(indices) + 1):
    selected_features = indices[:k]

    # 训练模型使用顶部 k 个特征
    rf_k = RandomForestRegressor(**grid_search.best_params_, random_state=42)
    rf_k.fit(X_train_scaled[:, selected_features], y_train)

    # 评估模型
    y_pred = rf_k.predict(X_test_scaled[:, selected_features])
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)


# 用于展示不同特征子集的MSE, MAE, R²的图
plt.figure(figsize=(14, 10))
plt.subplot(311)
plt.plot(mse_scores, label="MSE", color="r")
plt.title("MSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.subplot(312)
plt.plot(mae_scores, label="MAE", color="g")
plt.title("MAE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("Mean Absolute Error")
plt.legend()

plt.subplot(313)
plt.plot(r2_scores, label="R²", color="b")
plt.title("R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("R² Score")
plt.legend()

plt.tight_layout()
plt.show()

# 最佳模型
best_model = grid_search.best_estimator_

# 模型预测
y_pred = best_model.predict(X_test_scaled)

# 模型评估
y_pred = best_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)
