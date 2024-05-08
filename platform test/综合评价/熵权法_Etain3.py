import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# 读取数据
file_path = "E-data_train3.csv"
data = pd.read_csv(file_path)

# 删除'环境得分（华证）'列
data_dropped = data.drop("环境得分（华证）", axis=1)

# 处理缺失值：使用多重插补
imputer = IterativeImputer(random_state=0)
data_imputed = imputer.fit_transform(data_dropped.iloc[:, 1:])  # 忽略第一列'企业'名称

# 数据标准化
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_imputed)


def entropy_weight(data):
    # 计算概率
    epsilon = 1e-12
    p = data / data.sum(axis=0)
    p = np.where(p == 0, epsilon, p)  # 防止0值输入log函数

    # 计算熵
    entropy = -np.sum(p * np.log(p), axis=0) / np.log(len(data))

    # 计算权重
    weights = (1 - entropy) / (1 - entropy).sum()
    return weights


# 计算权重
weights = entropy_weight(data_normalized)

# 输出权重
column_names = data_dropped.columns[1:]  # 获取除了'企业'列之外的列名
weights_output = pd.DataFrame(weights, index=column_names, columns=["权重"])
print(weights_output)
