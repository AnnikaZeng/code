from keras.models import Sequential
from keras.layers import Dense


def create_model(input_dim):
    model = Sequential(
        [
            Dense(30, input_dim=input_dim, activation="relu"),
            Dense(20, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model
