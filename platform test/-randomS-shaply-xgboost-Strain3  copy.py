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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy import stats

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
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
model = xgb.XGBRegressor(objective="reg:squarederror")

# 设置要测试的不同超参数
param_dist = {
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6, 8, 10],
    "alpha": [1, 5, 10],
    "n_estimators": [100, 200],
    "subsample": [0.5, 0.7, 1.0],
    "min_child_weight": [1, 3, 5],
    "lambda": [1, 2, 3],
}

# 创建 RandomizedSearchCV 实例
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,  # 试验50个不同的参数组合
    scoring="neg_mean_squared_error",
    cv=3,  # 3折交叉验证
    verbose=2,
    random_state=42,
    n_jobs=-1,  # 使用所有可用的CPU核心
)

# 进行模型拟合
random_search.fit(X_train_scaled, y_train)

print("最佳参数：", random_search.best_params_)
print("最佳得分（负MSE）：", random_search.best_score_)

# 使用最佳模型进行预测
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)


# 模型评估
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)

# 使用 XGBoost 的 feature_importances_ 获取特征重要性
feature_importances = best_model.feature_importances_
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

# # 可视化特征重要性
# explainer = shap.TreeExplainer(best_model)
# shap_values = explainer.shap_values(X_test_scaled)
# shap.summary_plot(shap_values, X_test_scaled)

# # 比较实际值与预测值
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.75, color="red")
# plt.xlabel("实际")
# plt.ylabel("预测")
# plt.title("实际值 vs 预测值")
# plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
# plt.show()
