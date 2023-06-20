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
ouput = keras.layers.Dense(5, activation='linear')(x)