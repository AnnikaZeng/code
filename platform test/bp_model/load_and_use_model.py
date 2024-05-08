from keras.models import load_model

# 加载模型
model = load_model("environment_model.h5")

# 获取模型参数
model.summary()

# 获取各层的权重和偏置
for layer in model.layers:
    weights, biases = layer.get_weights()
    print(f"Layer: {layer.name}")
    print("Weights:\n", weights)
    print("Biases:\n", biases)
