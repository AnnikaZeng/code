import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy import stats

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("S-data-train1.csv")

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
columns_to_check = full_imputed_data.columns.drop(["社会维度得分"])
for column in columns_to_check:
    z_scores = np.abs(stats.zscore(full_imputed_data[column]))
    outliers = z_scores > 3
    # 用中位数替换异常值
    median_val = np.median(full_imputed_data.loc[~outliers, column])
    full_imputed_data.loc[outliers, column] = median_val

# 特征选择和目标变量
X = full_imputed_data.drop(
    columns=["社会维度得分", "净利润纳税率（%）", "研发总额占营业收入比例"]
)
y = full_imputed_data["社会维度得分"]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建XGBoost模型
xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

# 设置要调优的超参数
# parameters = {
#     "colsample_bytree": [0.3, 0.5, 0.7],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "max_depth": [3, 5, 7],
#     "alpha": [1, 5, 10],
#     "n_estimators": [50, 100, 200],
# }
param_grid = {
    "colsample_bytree": [0.7, 0.8, 0.9],  # 围绕最优值 0.8
    "learning_rate": [0.05, 0.1],  # 围绕最优值 0.05
    "max_depth": [3, 5, 7],  # 围绕最优值 6
    "alpha": [1, 5],  # 围绕最优值 5
    "n_estimators": [100, 200],  # 围绕最优值 100
    "subsample": [0.4, 0.5, 0.6],  # 围绕最优值 0.5
    "min_child_weight": [2, 3, 4],  # 围绕最优值 3
}

# 使用网格搜索进行调优
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,  # 3折交叉验证
    scoring="neg_mean_squared_error",  # 使用负均方误差作为评分标准
    verbose=1,  # 日志显示详细程度
    n_jobs=-1,  # 使用所有可用的CPU核心
)
grid_search.fit(X_train_scaled, y_train)

# 打印最优参数和得分
print("最佳参数：", grid_search.best_params_)
best_score = np.sqrt(-grid_search.best_score_)
print("最佳得分的RMSE：", best_score)

# 使用最佳参数的模型进行预测
model = grid_search.best_estimator_

# 预测
y_pred = model.predict(X_test_scaled)

# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)

# 使用 XGBoost 的 feature_importances_ 获取特征重要性
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

# 打印重要的特征
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
