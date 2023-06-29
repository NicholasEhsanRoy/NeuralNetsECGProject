import math

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def plot_histories(loss, val_loss, hyperparameter_values, std_loss, std_val_loss):
    plt.figure()
    plt.xticks([0,1,2,3,4], ['{:.0e}'.format(x) for x in hyperparameter_values])

    plt.errorbar([0, 1, 2, 3, 4], loss, yerr = std_loss, color = 'blue', capsize=5, elinewidth=3)
    plt.errorbar([0, 1, 2, 3, 4], val_loss, yerr = std_val_loss, color = 'orange', capsize=5)
    plt.title("Loss of the model")
    plt.legend(["loss", "val_loss"])
    plt.xlabel('alpha value')
    plt.ylabel('loss / validation loss')


def load_data():
    samples = np.load("samples_centered_max_samples.npy")
    targets = np.load("targets_centered_max_samples.npy")
    return samples, targets

X, y = load_data()
split = int(len(X)*0.8)
X_train, X_test = X[:split],X[split:]
y_train, y_test = y[:split],y[split:]
hyperparameter_values = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]
std_val_loss = []
std_loss =[]
evaluation = []
loss = []
val_loss = []

for hyperparameter_value in hyperparameter_values:
    print(hyperparameter_value)
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
    model.add(keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
    model.add(keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = "categorical_crossentropy",
    optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics=["categorical_accuracy"])
    history = model.fit(X_train, y_train, epochs=500, verbose = False, validation_split=0.2)
    std_val_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
    std_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
    loss.append(sum((history.history['loss'][-30:])) / 30)
    val_loss.append(sum((history.history['val_loss'][-30:])) / 30)


plot_histories(loss, val_loss, hyperparameter_values, std_loss, std_val_loss)
plt.show()

