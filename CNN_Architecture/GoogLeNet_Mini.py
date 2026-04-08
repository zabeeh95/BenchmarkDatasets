import tensorflow as tf
from tensorflow.keras import layers, Model


class MiniGoogLeNet:

    @staticmethod
    def conv_module(x, filters, kx, ky, strides, padding="same"):
        x = layers.Conv2D(filters, (kx, ky), strides=strides, padding=padding)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    @staticmethod
    def inception_module(x, f1, f3):
        # parallel paths
        conv_1x1 = MiniGoogLeNet.conv_module(x, f1, 1, 1, (1, 1))
        conv_3x3 = MiniGoogLeNet.conv_module(x, f3, 3, 3, (1, 1))

        # concatenate along channels
        x = layers.Concatenate(axis=-1)([conv_1x1, conv_3x3])
        return x

    @staticmethod
    def downsample_module(x, filters):
        conv_3x3 = MiniGoogLeNet.conv_module(
            x, filters, 3, 3, (2, 2), padding="valid"
        )
        pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = layers.Concatenate(axis=-1)([conv_3x3, pool])
        return x

    @staticmethod
    def build(width, height, depth, classes):
        inputs = tf.keras.Input(shape=(height, width, depth))

        x = MiniGoogLeNet.conv_module(inputs, 96, 3, 3, (1, 1))

        # Stage 1
        x = MiniGoogLeNet.inception_module(x, 32, 32)
        x = MiniGoogLeNet.inception_module(x, 32, 48)
        x = MiniGoogLeNet.downsample_module(x, 80)

        # Stage 2
        x = MiniGoogLeNet.inception_module(x, 112, 48)
        x = MiniGoogLeNet.inception_module(x, 96, 64)
        x = MiniGoogLeNet.inception_module(x, 80, 80)
        x = MiniGoogLeNet.inception_module(x, 48, 96)
        x = MiniGoogLeNet.downsample_module(x, 96)

        # Stage 3
        x = MiniGoogLeNet.inception_module(x, 176, 160)
        x = MiniGoogLeNet.inception_module(x, 176, 160)

        # Head
        x = layers.AveragePooling2D((7, 7))(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(classes, activation="softmax")(x)

        model = Model(inputs, outputs, name="MiniGoogLeNet")

        return model