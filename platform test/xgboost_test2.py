import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

# 加载数据
data = pd.read_csv("E-data_train1.csv")

# # 数据清洗：去除可能的空格和逗号，转换数据类型
# data = data.replace(r'\s+', '', regex=True)  # 去除空格
# data = data.replace(',', '', regex=True)  # 去除逗号

# # 转换列的数据类型
# numerical_cols = ['全年能源消耗总量（万吨标煤）', '耗电（万千瓦时）', '耗气', '燃油量', '万元营业收入综合能耗（吨标煤/万元）', '二氧化碳排放量（万吨）', '单位营业收入二氧化碳排放量（吨/万元）', '环境得分']
# for col in numerical_cols:
#     data[col] = pd.to_numeric(data[col], errors='coerce')

# 处理缺失值
data.fillna(data.mean(), inplace=True)

# 选择特征和目标变量
X = data.drop(columns=["企业", "环境得分（华证"])
y = data["环境得分（华证"]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建XGBoost模型
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.3,
    learning_rate=0.1,
    max_depth=5,
    alpha=10,
    n_estimators=100,
)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RMSE: ", rmse)
print("MAE: ", mae)
print("R²: ", r2)

# 特征重要性
importances = model.feature_importances_
features = X.columns
print("Feature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance}")
