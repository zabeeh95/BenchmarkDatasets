from tensorflow.keras import layers,models

def BetterShallowNet(height, width, channels, classes):
    inputs = layers.Input(shape=(height, width, channels))

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(classes, activation="softmax")(x)

    return models.Model(inputs, outputs)
