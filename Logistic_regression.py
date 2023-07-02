import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Loads the data from files into numpy arrays.
def load_data():
    samples = np.load("samples_centered_max_samples.npy")
    targets = np.load("targets_centered_max_samples.npy")
    return samples, targets
#Function to plot the average of the last 30 epochs of a model with error bars. Used to find the optimal
#regularization term.
def plot_histories(loss, val_loss, hyperparameter_values, std_loss, std_val_loss):
    plt.figure()
    plt.xticks([0,1,2,3,4], ['{:.0e}'.format(x) for x in hyperparameter_values])

    plt.errorbar([0,1,2,3,4], loss, yerr = std_loss, color = 'blue', capsize=5, elinewidth=3)
    plt.errorbar([0,1,2,3,4], val_loss, yerr = std_val_loss, color = 'orange', capsize=5)
    plt.title("Loss of the model")
    plt.legend(["loss", "val_loss"])
    plt.xlabel('alpha value')
    plt.ylabel('loss / validation loss')
#Splitting the data into a train and testset.
X, y = load_data()
split = int(len(X)*0.8)
X_train, X_test = X[:split],X[split:]
y_train, y_test = y[:split],y[split:]



checkpoint_path = "Saved_models/logistic_regression.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

#Commented code below was used for hyperparameter tuning.
alpha_values = [100, 10 , 1, 0.1, 0.01]
std_val_loss = []
std_loss =[]
evaluation = []
loss = []
val_loss = []

# for alpha in alpha_values:
#     print(alpha)
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(alpha)))
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
#     loss="categorical_crossentropy",
#     metrics=['categorical_accuracy'])
#     history = model.fit(X_train, y_train, epochs=500, verbose = False, validation_split=0.2)
#     std_val_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
#     std_loss.append(np.std(history.history['loss'][-30:])/math.sqrt(30))
#     loss.append(sum((history.history['loss'][-30:])) / 30)
#     val_loss.append(sum((history.history['val_loss'][-30:])) / 30)
#
# plot_histories(loss, val_loss, alpha_values, std_loss, std_val_loss)
# plt.show()




model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(1)))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, epochs=500, verbose=True,
              callbacks=[cp_callback])

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