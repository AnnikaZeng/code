import pandas as pd
from model_definition import create_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载数据，预处理
data = pd.read_csv("E-data_train1.csv")
# 数据处理省略...

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建模型
model = create_model(X_train.shape[1])

# 训练模型
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# 保存模型
model.save("environment_model.h5")
