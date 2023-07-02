import math

import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#Function to plot the average of the last 30 epochs of a model with error bars. Used to find the optimal
#regularization term.
def plot_histories(loss, val_loss, hyperparameter_values, std_loss, std_val_loss):
    plt.figure()
    plt.xticks([0, 1, 2, 3, 4, 5], ['{:.0e}'.format(x) for x in hyperparameter_values])

    plt.errorbar([0, 1, 2, 3, 4, 5], loss, yerr = std_loss, color = 'blue', capsize=5, elinewidth=3)
    plt.errorbar([0, 1, 2, 3, 4, 5], val_loss, yerr = std_val_loss, color = 'orange', capsize=5)
    plt.title("Loss of the model")
    plt.legend(["loss", "val_loss"])
    plt.xlabel('alpha value')
    plt.ylabel('loss / validation loss')

#Loads the data from files into numpy arrays.
def load_data():
    samples = np.load("samples_centered_max_samples.npy")
    targets = np.load("targets_centered_max_samples.npy")
    return samples, targets

#Splitting the data into a train and testset.
X, y = load_data()
split = int(len(X)*0.8)
X_train, X_test = X[:split],X[split:]
y_train, y_test = y[:split],y[split:]

#Commented code below was used for hyperparameter tuning.
# hyperparameter_values = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0]
# std_val_loss = []
# std_loss =[]
# evaluation = []
# loss = []
# val_loss = []
#
# for hyperparameter_value in hyperparameter_values:
#     print(hyperparameter_value)
#     model = keras.Sequential()
#     model.add(keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
#     model.add(keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
#     model.add(keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(hyperparameter_value)))
#     model.add(keras.layers.Dense(5, activation='softmax'))
#     model.compile(loss = "categorical_crossentropy",
#     optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics=["categorical_accuracy"])
#     history = model.fit(X_train, y_train, epochs=500, verbose = False, validation_split=0.2)
#     std_val_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
#     std_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
#     loss.append(sum((history.history['loss'][-30:])) / 30)
#     val_loss.append(sum((history.history['val_loss'][-30:])) / 30)
#
#
# plot_histories(loss, val_loss, hyperparameter_values, std_loss, std_val_loss)
# plt.show()

checkpoint_path = "Saved_models/model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
#Training of the final model with all of the training data and the optimal regularizing term.
optimal_alpha = 0.000001
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(optimal_alpha)))
model.add(keras.layers.Dense(48, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(optimal_alpha)))
model.add(keras.layers.Dense(32, activation='sigmoid', kernel_regularizer=keras.regularizers.L2(optimal_alpha)))
model.add(keras.layers.Dense(5, activation='softmax'))
model.compile(loss = "categorical_crossentropy",
optimizer = keras.optimizers.Adam(learning_rate = 0.001), metrics=["categorical_accuracy"])

history = model.fit(X_train, y_train, epochs=500, callbacks=[cp_callback], verbose = 1)





#A set of predictions made on the test set by the model are obtained and compared to the ground truth.
#The the outcome is then plotted in a confusion matrix
y_pred = model.predict(X_test)
y_true = y_test
matrix = multilabel_confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

labels = ["N", "L", "R", "V", "/"]

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)


plt.figure()
plt.plot(history.history["loss"], color='blue')
plt.legend(["loss"])
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.show()