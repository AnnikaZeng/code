import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

# 创建一个包含随机数据的DataFrame
np.random.seed(42)  # 为了重复性的随机结果
data = np.random.rand(100, 5)  # 100行5列的随机数据
columns = ["feature1", "feature2", "feature3", "feature4", "target"]  # 列名
df = pd.DataFrame(data, columns=columns)

# 将目标变量转换为二进制分类
df["target"] = (df["target"] > 0.5).astype(int)

# 根据原有代码逻辑分割特征和目标变量
X = df.drop("target", axis=1)
y = df["target"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练XGBoost模型
model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
model.fit(X_train, y_train)

# 创建SHAP解释器
explainer = shap.TreeExplainer(model)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 计算每个特征的平均绝对SHAP值
shap_sum = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame(
    list(zip(X.columns, shap_sum)), columns=["feature", "shap_importance"]
)
feature_importance.sort_values(by="shap_importance", ascending=False, inplace=True)

# 输出特征重要性
print(feature_importance)

# 可视化特征重要性
shap.summary_plot(shap_values, X_test)
