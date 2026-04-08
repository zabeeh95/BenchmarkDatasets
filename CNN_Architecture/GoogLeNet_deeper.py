import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class DeeperGoogLeNet:

    @staticmethod
    def conv_module(x, filters, kx, ky, strides, padding="same", reg=0.0005, name=None):
        x = layers.Conv2D(filters, (kx, ky),
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=regularizers.l2(reg),
                          name=None if name is None else name + "_conv")(x)

        x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)

        x = layers.ReLU(name=None if name is None else name + "_relu")(x)

        return x

    @staticmethod
    def inception_module(x, f1, f3_reduce, f3, f5_reduce, f5, f_pool, stage, reg=0.0005):
        # 1x1 branch
        b1 = DeeperGoogLeNet.conv_module(
            x, f1, 1, 1, (1, 1), reg=reg, name=stage + "_b1")

        # 1x1 → 3x3 branch
        b2 = DeeperGoogLeNet.conv_module(
            x, f3_reduce, 1, 1, (1, 1), reg=reg, name=stage + "_b2_1")
        b2 = DeeperGoogLeNet.conv_module(
            b2, f3, 3, 3, (1, 1), reg=reg, name=stage + "_b2_2")

        # 1x1 → 5x5 branch
        b3 = DeeperGoogLeNet.conv_module(
            x, f5_reduce, 1, 1, (1, 1), reg=reg, name=stage + "_b3_1")
        b3 = DeeperGoogLeNet.conv_module(
            b3, f5, 5, 5, (1, 1), reg=reg, name=stage + "_b3_2")

        # Pool → 1x1 branch
        b4 = layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding="same",
                                 name=stage + "_pool")(x)
        b4 = DeeperGoogLeNet.conv_module(
            b4, f_pool, 1, 1, (1, 1), reg=reg, name=stage + "_b4")

        # Concatenate
        x = layers.Concatenate(name=stage + "_concat")([b1, b2, b3, b4])

        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0005):
        inputs = tf.keras.Input(shape=(height, width, depth))

        # Initial layers
        x = DeeperGoogLeNet.conv_module(inputs, 64, 5, 5, (1, 1), reg=reg, name="block1")
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        x = DeeperGoogLeNet.conv_module(x, 64, 1, 1, (1, 1), reg=reg, name="block2")
        x = DeeperGoogLeNet.conv_module(x, 192, 3, 3, (1, 1), reg=reg, name="block3")
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # Inception blocks
        x = DeeperGoogLeNet.inception_module(x, 64, 96, 128, 16, 32, 32, "3a")
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 192, 32, 96, 64, "3b")
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        x = DeeperGoogLeNet.inception_module(x, 192, 96, 208, 16, 48, 64, "4a")
        x = DeeperGoogLeNet.inception_module(x, 160, 112, 224, 24, 64, 64, "4b")
        x = DeeperGoogLeNet.inception_module(x, 128, 128, 256, 24, 64, 64, "4c")
        x = DeeperGoogLeNet.inception_module(x, 112, 144, 288, 32, 64, 64, "4d")
        x = DeeperGoogLeNet.inception_module(x, 256, 160, 320, 32, 128, 128, "4e")
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        # Final layers
        x = layers.AveragePooling2D((4, 4))(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(classes, activation="softmax")(x)

        model = Model(inputs, outputs, name="GoogLeNet")

        return model
