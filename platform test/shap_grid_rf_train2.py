import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据
data = pd.read_csv("E-data_train2.csv")

# 使用多重插补方法处理缺失值
imputer = IterativeImputer(
    estimator=RandomForestRegressor(),
    missing_values=np.nan,
    max_iter=20,
    random_state=42,
    tol=0.001,  # 可以调整收敛容忍度，默认是1e-3，调小这个值会使得收敛条件更严格
)
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

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

# 设置要测试的参数网格
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

# 创建随机森林模型
rf = RandomForestRegressor(random_state=42)

# 设置网格搜索
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    scoring="neg_mean_squared_error",
)

# 执行网格搜索
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数创建随机森林模型
best_rf = RandomForestRegressor(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train_scaled, y_train)

# 预测
rf_y_pred = best_rf.predict(X_test_scaled)

# 模型评估
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_mae = mean_absolute_error(y_test, rf_y_pred)
rf_r2 = r2_score(y_test, rf_y_pred)
print("Random Forest - RMSE: ", rf_rmse)
print("Random Forest - MAE: ", rf_mae)
print("Random Forest - R²: ", rf_r2)

# 特征重要性
rf_importances = best_rf.feature_importances_
features = X.columns
sorted_indices = np.argsort(rf_importances)[::-1]
sorted_features = features[sorted_indices]
sorted_importances = rf_importances[sorted_indices]

print("Random Forest - Feature Importances:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance:.4f}")

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, sorted_importances, color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest - Sorted Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# 创建SHAP解释器并计算SHAP值（随机森林）
rf_explainer = shap.TreeExplainer(best_rf)
rf_shap_values = rf_explainer.shap_values(X_test_scaled)

# 可视化特征重要性（随机森林）
shap.summary_plot(rf_shap_values, X_test_scaled, feature_names=X.columns)
