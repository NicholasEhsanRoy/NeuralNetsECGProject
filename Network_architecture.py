import numpy as np
import tensorflow as tf
import keras

def load_data():
    samples = np.load("samples.npy")
    targets = np.load("targets.npy")
    return samples, targets


X, y = load_data()


input = keras.Input(shape=(144,))
x = keras.layers.Conv(64, activation='sigmoid')(inputs)
x = keras.layers.Conv(64, activation='sigmoid')(x)
x = keras.layers.Conv(32, activation='sigmoid')(x)
x = keras.layers.Conv(32, activation='sigmoid')(x)
x = keras.layers.Conv(32, activation='sigmoid')(x)
x = keras.layers.Dense(16, activation='sigmoid')(x)
ouput = keras.layers.Dense(5, activation='linear')(x)

model = keras.Model(input, output)

model.compile(loss = "MSE",
optimizer = optimizers.Adam(learning_rate = 0.001))

history = model.fit(samples, targets, epochs=1000, verbose=True)

print(history)
