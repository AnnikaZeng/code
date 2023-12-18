# 导入数据分析三大神器

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")  # 忽略警告

# 设定角度
angles = np.linspace(0, 2 * np.pi, 7, endpoint=True)  #

# 设置数据
data = np.array([[7] * 7])  # 这里设置7个数据，并且保证第一和最后一个数据相等

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, polar=True)  # 使用极坐标

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像是负号'-'显示为方块的问题

# 画线
# for i in range():
ax.plot(angles, data[0], linewidth=2)  # linewidth=2线条宽度

# 填充颜色，alpha表示透明高度
ax.fill(angles, data[0], alpha=0.5, color="red")

# 画轴并画出轴的标签，fontsize设置字体大小
ax.set_thetagrids(
    angles[:-1] * 180 / np.pi, ["力量", "速度", "技巧", "发球", "防守", "经验"], fontsize=15
)

# 添加标题
plt.title("马龙实力雷达图", fontsize=20)

# 添加网格线
ax.grid(True)


# 去掉最外围的黑圈
ax.spines["polar"].set_visible(False)

# 显示图形
plt.show()
