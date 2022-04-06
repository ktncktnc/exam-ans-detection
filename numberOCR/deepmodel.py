import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization


def load_nnet_model(checkpoint=None):
    nnet_model = Sequential()

    nnet_model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)))
    nnet_model.add(Conv2D(64, 3, activation='relu', padding='same'))
    nnet_model.add(MaxPooling2D(2, 2))
    nnet_model.add(BatchNormalization())

    nnet_model.add(Conv2D(128, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(128, 3, activation='relu', padding='same'))
    nnet_model.add(MaxPooling2D(2, 2))
    nnet_model.add(BatchNormalization())

    nnet_model.add(Conv2D(256, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(256, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(256, 3, activation='relu', padding='same'))
    nnet_model.add(MaxPooling2D(2, 2))
    nnet_model.add(BatchNormalization())

    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(MaxPooling2D(2, 1))  # default stride is 2
    nnet_model.add(BatchNormalization())

    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    nnet_model.add(MaxPooling2D(2, 1))  # default stride is 2
    nnet_model.add(BatchNormalization())

    nnet_model.add(Flatten())
    nnet_model.add(Dense(4096, activation='relu'))
    nnet_model.add(Dropout(0.5))
    nnet_model.add(Dense(4096, activation='relu'))
    nnet_model.add(Dropout(0.5))
    nnet_model.add(Dense(10, activation='softmax'))

    if checkpoint is not None:
        nnet_model.load_weights(checkpoint)

    return nnet_model


def predict(model, imgs):
    result = model.predict(imgs)

    return np.argmax(result, axis=1), np.max(result, axis=1)