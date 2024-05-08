import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer


# 加载数据
data = pd.read_csv("E-data_train3.csv")

# 假设 '企业' 是包含唯一标识符或非相关文本数据的列
# 在插补过程中排除这一列
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

# 对任何剩余的缺失值应用多重插补
iterative_imputer = IterativeImputer(max_iter=10, random_state=0)
full_imputed_data = iterative_imputer.fit_transform(knn_imputed_data)

# 将完全插补后的数据转换回 DataFrame
full_imputed_data = pd.DataFrame(full_imputed_data, columns=data_to_impute.columns)

# 将被排除的 '企业' 列重新添加到完全插补后的数据集中
full_imputed_data["企业"] = data["企业"]

# 异常值检验和输出具体值
columns_to_check = full_imputed_data.columns.drop(["企业", "环境得分（华证）"])
for column in columns_to_check:
    if pd.api.types.is_numeric_dtype(full_imputed_data[column]):
        z_scores = np.abs(stats.zscore(full_imputed_data[column]))
        outliers = z_scores > 3
        if np.any(outliers):
            print(f"{column} 的异常值:")
            print(full_imputed_data.loc[outliers, column])
