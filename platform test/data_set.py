import pandas as pd
import numpy as np

# 创建示例数据集
np.random.seed(0)
data = pd.DataFrame(
    {
        "Company": ["Company A", "Company B", "Company C", "Company D", "Company E"],
        "Water_Use": np.random.randint(1, 100, size=5),
        "Energy_Consumption": np.random.randint(1, 100, size=5),
        "Waste_Management": np.random.randint(1, 100, size=5),
        "ESG_score": np.random.randint(1, 100, size=5),
    }
)

# 将数据集保存到CSV文件
data.to_csv("esg_data.csv", index=False)

print("示例数据集已创建并保存到文件 esg_data.csv")
