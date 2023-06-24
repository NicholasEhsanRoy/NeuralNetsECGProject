import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_data():
    samples = np.load("samples_centered_max_samples.npy")
    targets = np.load("targets_centered_max_samples.npy")
    return samples, targets

X, y = load_data()
split = int(len(X)*0.8)
X_train, X_test = X[:split],X[split:]
y_train, y_test = y[:split],y[split:]
alpha_values = [0.00001]
evaluation = []
for alpha in alpha_values:
    print(alpha)
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(alpha)))
    model.add(keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(alpha)))
    model.add(keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(alpha)))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile(loss = "categorical_crossentropy",
    optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics=["categorical_accuracy"])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5,mode ='min')
    model.fit(X_train, y_train, epochs=100, verbose = False, callbacks=[callback], validation_split=0.2)
    evaluation.append([model.evaluate(X_test,y_test),alpha])
print(evaluation)
