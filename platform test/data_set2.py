import pandas as pd
import numpy as np

# 设置随机种子以复现结果
np.random.seed(42)

# 创建数据字典
data = {
    "resource_consumption": np.random.normal(
        loc=100, scale=20, size=100
    ),  # 均值为100，标准差为20
    "energy_consumption": np.random.normal(
        loc=200, scale=50, size=100
    ),  # 均值为200，标准差为50
    "carbon_emission": np.random.normal(
        loc=150, scale=30, size=100
    ),  # 均值为150，标准差为30
    "esg_score": np.random.normal(
        loc=50, scale=15, size=100
    ),  # ESG评分，均值为50，标准差为15
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 将数据集写入CSV文件
df.to_csv("example_data.csv", index=False)
