import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import export_graphviz
import graphviz

# 加载数据集
iris = load_iris()
X = iris["data"]  # Equivalent to iris.data
y = iris["target"]  # Equivalent to iris.target


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树模型
tree_model = DecisionTreeClassifier(random_state=42)

# 训练模型
tree_model.fit(X_train, y_train)

# 进行预测
y_pred = tree_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 可视化决策树（需要安装 graphviz 库）

dot_data = export_graphviz(
    tree_model,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True,
)
graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")
