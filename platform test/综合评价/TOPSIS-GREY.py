import numpy as np


def topsis_grey(data, weights):
    # 正向化处理（确保更大的值是更好的）
    normalized_data = data / np.max(data, axis=0)

    # 计算权重化的标准化决策矩阵
    weighted_normalized = normalized_data * weights

    # 确定正理想解和负理想解
    ideal_positive = np.max(weighted_normalized, axis=0)
    ideal_negative = np.min(weighted_normalized, axis=0)

    # 计算与正理想解和负理想解的距离
    distance_positive = np.sqrt(
        np.sum((weighted_normalized - ideal_positive) ** 2, axis=1)
    )
    distance_negative = np.sqrt(
        np.sum((weighted_normalized - ideal_negative) ** 2, axis=1)
    )

    # 计算相对贴近度
    relative_closeness = distance_negative / (distance_positive + distance_negative)

    # 计算各维度的相对贴近度
    dimension_closeness = []
    for i in range(data.shape[1]):
        dim_weighted_norm = normalized_data[:, i] * weights[i]
        dim_pos = np.max(dim_weighted_norm)
        dim_neg = np.min(dim_weighted_norm)
        dim_dist_pos = np.sqrt((dim_weighted_norm - dim_pos) ** 2)
        dim_dist_neg = np.sqrt((dim_weighted_norm - dim_neg) ** 2)
        dim_closeness = dim_dist_neg / (dim_dist_pos + dim_dist_neg)
        dimension_closeness.append(dim_closeness)

    return relative_closeness, dimension_closeness


# 示例数据
data = np.array(
    [
        [0.8, 0.7, 0.9],  # 方案1
        [0.6, 0.9, 0.5],  # 方案2
        [0.9, 0.6, 0.8],  # 方案3
    ]
)
weights = np.array([0.3, 0.5, 0.2])  # 权重

# 计算
overall_closeness, dimensions_closeness = topsis_grey(data, weights)

print("总体相对贴近度:", overall_closeness)
for idx, dc in enumerate(dimensions_closeness, start=1):
    print(f"维度{idx}的相对贴近度:", dc)
