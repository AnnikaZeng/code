import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 定义评分的范围
x = np.arange(0, 101, 1)

# 定义隶属度函数
mfx_poor = fuzz.trapmf(x, [0, 0, 25, 60])  # 升梯形函数
mfx_average = fuzz.trimf(x, [25, 60, 80])  # 三角形函数，覆盖中等评分
mfx_good = fuzz.trimf(x, [60, 80, 95])  # 三角形函数，覆盖良好评分
mfx_excellent = fuzz.trapmf(x, [80, 95, 100, 100])  # 降梯形函数

# 评分数据
scores = {
    "Reliability": 82,
    "Cost-effectiveness": 67,
    "Time Management": 45,
    "Customer Satisfaction": 90,
}

# 计算隶属度
membership_degrees = {}
for key, value in scores.items():
    degrees = {
        "Poor": fuzz.interp_membership(x, mfx_poor, value),
        "Average": fuzz.interp_membership(x, mfx_average, value),
        "Good": fuzz.interp_membership(x, mfx_good, value),
        "Excellent": fuzz.interp_membership(x, mfx_excellent, value),
    }
    membership_degrees[key] = degrees
    print(f"{key}: {degrees}")

# 绘制隶属度函数图形
plt.figure(figsize=(10, 5))
plt.plot(x, mfx_poor, "y", linewidth=1.5, label="Poor")
plt.plot(x, mfx_average, "r", linewidth=1.5, label="Average")
plt.plot(x, mfx_good, "g", linewidth=1.5, label="Good")
plt.plot(x, mfx_excellent, "b", linewidth=1.5, label="Excellent")
plt.title("Membership Functions for Project Performance Evaluation")
plt.ylabel("Membership Degree")
plt.xlabel("Score")
plt.legend()
plt.ylim(-0.1, 1.1)
plt.show()
