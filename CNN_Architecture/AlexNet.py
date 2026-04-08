import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_alexnet(height, width , channel, classes, reg=0.0002):
    input_shape = (height, width, channel)
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    # Block 1
    x = layers.Conv2D(96, (11, 11), strides=4, padding="same",
                      kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(256, (5, 5), padding="same",
                      kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(384, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(384, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(256, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2)(x)
    x = layers.Dropout(0.25)(x)

    # Fully Connected Layers
    x = layers.Flatten()(x)

    x = layers.Dense(4096, kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(4096, kernel_regularizer=regularizers.l2(reg))(x)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Output
    outputs = layers.Dense(classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="AlexNet")

    return model