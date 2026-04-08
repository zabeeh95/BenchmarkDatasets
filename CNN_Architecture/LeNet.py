import tensorflow as tf
from tensorflow.keras import layers, models


def build_lenet(width, height, depth, classes):
    inputs = tf.keras.Input(shape=(height, width, depth))

    x = inputs

    # Block 1
    x = layers.Conv2D(20, (5, 5), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Block 2
    x = layers.Conv2D(50, (5, 5), padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    # Fully Connected
    x = layers.Flatten()(x)
    x = layers.Dense(500)(x)
    x = layers.ReLU()(x)

    # Output
    outputs = layers.Dense(classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="LeNet")

    return model