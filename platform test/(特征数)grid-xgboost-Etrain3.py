import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import statsmodels.api as sm
from scipy import stats

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.family"] = "Microsoft YaHei"  # 微软雅黑
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("E-data_train3+.csv")

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

# 异常值处理
columns_to_check = full_imputed_data.columns.drop(
    ["环境得分（华证）"]
)  # 排除“环境得分（华证）”
for column in columns_to_check:
    z_scores = np.abs(stats.zscore(full_imputed_data[column]))
    outliers = z_scores > 3
    # 用中位数替换异常值
    median_val = np.median(full_imputed_data.loc[~outliers, column])
    full_imputed_data.loc[outliers, column] = median_val

# 特征选择和目标变量
X = full_imputed_data.drop(
    columns=[
        "环境得分（华证）",
        "万元营业收入综合能耗（吨标煤/万元）",
        "单位营业收入二氧化碳排放量（吨/万元）",
        "能耗变幅（%）",
    ]
)
y = full_imputed_data["环境得分（华证）"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 XGBoost 模型
model = xgb.XGBRegressor(objective="reg:squarederror")

# 设置要测试的不同超参数
param_grid = {
    "colsample_bytree": [0.3, 0.7],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "alpha": [5, 10],
    "n_estimators": [100, 200],
}

# 网格搜索调优
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    verbose=1,
)
grid_search.fit(X_train_scaled, y_train)

# 最佳参数和模型
best_model = grid_search.best_estimator_
print("最佳参数：", grid_search.best_params_)

# 使用最佳模型进行预测
best_model.fit(X_train_scaled, y_train)  # 确保最佳模型已经拟合数据
y_pred = best_model.predict(X_test_scaled)

# 获取特征重要性并排序其索引
feature_importances = best_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

# 打印重要的特征，确保 sorted_idx 正确生成
print("特征排序：")
for index in sorted_idx:
    print(f"{X.columns[index]}: {feature_importances[index]}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(X.columns[sorted_idx], feature_importances[sorted_idx], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# 现在使用sorted_idx来进行特征子集选择和模型性能的评估
# 使用GridSearchCV找到的最佳模型参数
best_params = grid_search.best_params_
print("最优参数组合:", best_params)

# # 获取并打印最优参数组合的最佳平均分数（注意: 转换为正的RMSE）
# best_score = np.sqrt(-grid_search.best_score_)
# print("对应的最佳平均RMSE:", best_score)

# # 获取并打印最优参数在各个交叉验证折中的分数
# best_index = grid_search.best_index_
# cv_results = grid_search.cv_results_

# print("\n最优参数在各折交叉验证中的详细得分:")
# for fold_index in range(grid_search.cv):
#     key = f"split{fold_index}_test_score"
#     fold_score = np.sqrt(-cv_results[key][best_index])
#     print(f"折{fold_index + 1}的RMSE: {fold_score}")

# 特征子集选择和模型性能评估
results = []
max_features = len(X_train_scaled[0])

for num_features in range(1, max_features + 1):
    selected_features_indices = sorted_idx[:num_features]
    X_train_subset = X_train_scaled[:, selected_features_indices]
    X_test_subset = X_test_scaled[:, selected_features_indices]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=best_params["colsample_bytree"],
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        alpha=best_params["alpha"],
        n_estimators=best_params["n_estimators"],
    )

    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_test_subset)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((num_features, rmse, mae, r2))

# 解包结果以用于绘图
features_count, rmses, maes, r2s = zip(*results)

# 绘图查看特征数量与模型性能的关系
plt.figure(figsize=(14, 7))

plt.subplot(1, 3, 1)
plt.plot(features_count, rmses, marker="o")
plt.title("RMSE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("RMSE")

plt.subplot(1, 3, 2)
plt.plot(features_count, maes, marker="o")
plt.title("MAE vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("MAE")

plt.subplot(1, 3, 3)
plt.plot(features_count, r2s, marker="o")
plt.title("R² vs. Number of Features")
plt.xlabel("Number of Features")
plt.ylabel("R²")

plt.tight_layout()
plt.show()


# 找出最佳的模型性能
min_rmse = min(rmses)
min_mae = min(maes)
max_r2 = max(r2s)

# 对应的特征数量
best_rmse_count = features_count[rmses.index(min_rmse)]
best_mae_count = features_count[maes.index(min_mae)]
best_r2_count = features_count[r2s.index(max_r2)]

# 输出最佳性能指标及对应的特征数量
print("最佳性能评估：")
print(f"最小RMSE: {min_rmse}, 特征数量: {best_rmse_count}")
print(f"最小MAE: {min_mae}, 特征数量: {best_mae_count}")
print(f"最大R²: {max_r2}, 特征数量: {best_r2_count}")

# 如果需要，也可以打印出这些特征的名称
print("对应的最佳特征（按R²）:")
best_features_r2 = [X.columns[idx] for idx in sorted_idx[:best_r2_count]]
print(best_features_r2)
