import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score


# 设置 Matplotlib 支持中文显示
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 'SimHei' 是一种常用的中文黑体
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 加载数据
data = pd.read_csv("E-data_train1.csv")
data.drop(columns=["Unnamed: 12"], inplace=True)

# 计算仅数值类型列的均值
numeric_columns = data.select_dtypes(include=[np.number]).columns
data_numeric_mean = data[numeric_columns].mean()

# 仅在数值类型的列上填充缺失值
data[numeric_columns] = data[numeric_columns].fillna(data_numeric_mean)

# 数据标准化处理
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# 选择特征和目标变量
X = data.drop(columns=["企业", "环境得分（华证）"])
y = data["环境得分（华证）"]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建模型
model = Sequential(
    [
        Dense(30, input_dim=X_train.shape[1], activation="relu"),
        Dense(20, activation="relu"),
        Dense(1, activation="linear"),
    ]
)

# 编译模型
model.compile(optimizer="adam", loss="mse")

# 早停机制
early_stopping_monitor = EarlyStopping(
    monitor="val_loss", patience=10, verbose=1, restore_best_weights=True
)

# 训练模型并保存训练过程
history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=10,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping_monitor],
)

# 预测
y_pred = model.predict(X_test).flatten()

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE: ", mse)
print("R²: ", r2)

# 绘制训练损失和验证损失
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 绘制实际值与预测值的散点图
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=2)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted Values")
plt.show()


model.summary()

# for layer in model.layers:
#     weights, biases = layer.get_weights()
#     print(f"Layer: {layer.name}")
#     print("Weights:\n", weights)
#     print("Biases:\n", biases)
#     print("Number of weights:", weights.size)
#     print("Number of biases:", biases.size)
