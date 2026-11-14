
from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(216, 1), num_classes=4):

    model = keras.Sequential([

        # Input
        layers.Input(shape=input_shape),

        # First convolution block
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Second convolution block
        layers.Conv1D(
            filters=64,
            kernel_size=5,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Third convolution block
        layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),

        # Convert extracted features into a vector
        layers.GlobalAveragePooling1D(),

        # Classification layers
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),

        # Four class probabilities
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()
