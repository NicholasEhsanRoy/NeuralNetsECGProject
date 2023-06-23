import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_data():
    samples = np.load("samples_centered.npy")
    targets = np.load("targets_centered.npy")
    return samples, targets

X, y = load_data()
X_train, X_test = X[0:8000],X[8000:]
y_train, y_test = y[0:8000],y[8000:]
# X.reshape(-1,144,1)
# y.reshape(-1,144,1)

# X = np.array([X])
# y = np.array([y])

print(X_test.shape)
print(y_train.shape)
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='sigmoid'))
model.add(keras.layers.Dense(48, activation='sigmoid'))
model.add(keras.layers.Dense(32, activation='sigmoid'))
model.add(keras.layers.Dense(5, activation='softmax'))
model.compile(loss = "categorical_crossentropy",
optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics=["categorical_accuracy"])
print(X.shape)
model.fit(X_train, y_train, epochs=30, verbose=True)
print(model.evaluate(X_test,y_test))
