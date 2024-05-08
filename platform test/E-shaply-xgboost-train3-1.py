import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import shap


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(columns=["企业"])
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].astype(str))
            label_encoders[column] = le
    return data, label_encoders


def impute_data(data):
    knn_imputer = KNNImputer(n_neighbors=5)
    data_knn = knn_imputer.fit_transform(data)

    iterative_imputer = IterativeImputer(
        estimator=RandomForestRegressor(), max_iter=10, random_state=42
    )
    data_imputed = iterative_imputer.fit_transform(data_knn)
    return pd.DataFrame(data_imputed, columns=data.columns)


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.1,
        max_depth=5,
        alpha=10,
        n_estimators=100,
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test_scaled, y_test, y_pred, rmse, mae, r2


def main():
    # 设置中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载和预处理数据
    data, label_encoders = load_and_preprocess_data("E-data_train3.csv")

    # 插补数据
    data_imputed = impute_data(data)

    # 分离特征和目标变量
    X = data_imputed.drop(columns=["环境得分（华证）"])
    y = data_imputed["环境得分（华证）"]

    # 训练模型并评估
    model, X_test_scaled, y_test, y_pred, rmse, mae, r2 = train_model(X, y)

    # 输出评估结果
    print("均方根误差: ", rmse)
    print("平均绝对误差: ", mae)
    print("R²: ", r2)

    # 解释模型并可视化
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test_scaled)


if __name__ == "__main__":
    main()
