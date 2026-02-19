from keras.datasets import mnist
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD


import json
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import Callback


class TrainingMonitorLive(Callback):
    def __init__(self, figPath=None, jsonPath=None):
        super().__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.H = {}

        plt.ion()  # interactive mode ON

    def on_train_begin(self, logs=None):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        if self.jsonPath and os.path.exists(self.jsonPath):
            self.H = json.loads(open(self.jsonPath).read())

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Update history
        for k, v in logs.items():
            self.H.setdefault(k, []).append(v)

        # Save JSON
        if self.jsonPath:
            with open(self.jsonPath, "w") as f:
                json.dump(self.H, f)

        # Clear and re-plot
        self.ax.clear()
        N = np.arange(len(self.H["loss"]))

        self.ax.plot(N, self.H["loss"], label="train_loss")
        self.ax.plot(N, self.H["val_loss"], label="val_loss")
        self.ax.plot(N, self.H["accuracy"], label="train_accuracy")
        self.ax.plot(N, self.H["val_accuracy"], label="val_accuracy")

        self.ax.set_title(f"Training Monitor (Epoch {epoch + 1})")
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss / Accuracy")
        self.ax.legend()
        self.ax.grid(alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Optional save
        if self.figPath:
            self.fig.savefig(self.figPath)


(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.astype("float") / 255.0
print(trainX.shape)
testX = testX.astype("float") / 255.0

trainX = trainX.reshape((trainX.shape[0], 784))
print(trainX.shape)
testX = testX.reshape((testX.shape[0], 784))

model = Sequential()
model.add(Dense(128, input_shape=(784,), activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer=SGD(0.01), metrics=["accuracy"])



callbacks = [TrainingMonitorLive(figPath="training_monitor.png", jsonPath="training_monitor.json")]

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=callbacks, verbose=1)
