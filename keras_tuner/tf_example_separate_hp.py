from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
import numpy as np

def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


if __name__ == "__main__":

    (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x[:-10000]
    x_val = x[-10000:]
    y_train = y[:-10000]
    y_val = y[-10000:]

    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    tuner = keras_tuner.RandomSearch(
                                    hypermodel=build_model,
                                    objective="val_accuracy",
                                    max_trials=3,
                                    executions_per_trial=2,
                                    overwrite=True,
                                    directory="my_dir",
                                    project_name="helloworld_02",)


    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

    print("********************************************************************")

    # Get the top 2 hyperparameters.
    best_hps = tuner.get_best_hyperparameters(5)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    # Fit with the entire dataset.
    x_all = np.concatenate((x_train, x_val))
    y_all = np.concatenate((y_train, y_val))
    model.fit(x=x_all, y=y_all, epochs=5)