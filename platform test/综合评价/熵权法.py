import numpy as np


def entropy_weight(data):
    # 归一化处理
    data_normalized = data / data.sum(axis=0)

    # 计算每个指标的熵值
    epsilon = 1e-12  # 避免对数为负无穷
    entropy = -np.sum(
        data_normalized * np.log(data_normalized + epsilon), axis=0
    ) / np.log(len(data))

    # 计算权重
    weights = (1 - entropy) / (1 - entropy).sum()

    return weights


# 示例数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 计算权重
weights = entropy_weight(data)
print("权重:", weights)
