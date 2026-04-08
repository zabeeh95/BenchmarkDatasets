import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


# ✅ Residual Block
def residual_block(x, filters, stride=1, reduce=False, reg=1e-4, bn_eps=2e-5, bn_mom=0.9):
    shortcut = x

    # 1x1
    y = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_mom)(x)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(int(filters * 0.25), (1, 1),
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(y)

    # 3x3
    y = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_mom)(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(int(filters * 0.25), (3, 3),
                      strides=stride,
                      padding="same",
                      use_bias=False,
                      kernel_regularizer=regularizers.l2(reg))(y)

    # 1x1
    y = layers.BatchNormalization(epsilon=bn_eps, momentum=bn_mom)(y)
    y = layers.Activation("relu")(y)
    y = layers.Conv2D(filters, (1, 1), use_bias=False, kernel_regularizer=regularizers.l2(reg))(y)

    #  Shortcut adjustment
    if reduce:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, use_bias=False,
                                 kernel_regularizer=regularizers.l2(reg))(x)

    # Add
    out = layers.Add()([y, shortcut])
    return out


def ResNet(width, height, depth, classes,
           stages, filters,
           reg=1e-4, dataset="cifar"):
    inputs = layers.Input(shape=(height, width, depth))
    x = layers.BatchNormalization()(inputs)

    # Initial layer
    if dataset == "cifar":
        x = layers.Conv2D(filters[0], (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizers.l2(reg))(
            x)

    elif dataset == "tiny_imagenet":
        x = layers.Conv2D(filters[0], (5, 5), padding="same", use_bias=False, kernel_regularizer=regularizers.l2(reg)) \
            (x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D((1, 1))(x)
        x = layers.MaxPooling2D((3, 3), strides=2)(x)

    for i in range(len(stages)):
        stride = 1 if i == 0 else 2

        # First block (with reduction)
        x = residual_block(x, filters[i + 1],
                           stride=stride,
                           reduce=True)

        # Remaining blocks
        for _ in range(stages[i] - 1):
            x = residual_block(x, filters[i + 1])

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="ResNet")
    return model
