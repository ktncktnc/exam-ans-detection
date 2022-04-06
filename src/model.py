from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, BatchNormalization
from src.dataset import *
import math
from sklearn.metrics import confusion_matrix
from keras.optimizer_v2.adam import Adam as Adam
from sklearn.model_selection import train_test_split

def load_nnet_model(checkpoint=None):
    model = Sequential()

    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 128, 1)))
    model.add(Conv2D(64, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    # nnet_model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    # nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Conv2D(1024, 3, activation='relu', padding='same'))
    # nnet_model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(1024, 3, activation='relu', padding='same'))
    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    if checkpoint is not None:
        model.load_weights(checkpoint).expect_partial()

    return model


def lr_schedule(epoch, lr, min_lr=0.0005):
    if epoch > 10:
        lr = max(min_lr, lr * math.exp(-0.05))

    return lr


def main():
    model = load_nnet_model()

    opt = Adam(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    X, y = load_dataset("../all/dataset.csv", "../all/tmp/")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath="../all/trained_model/cp.ckpt", save_weights_only=True, save_best_only=True, verbose=1)
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    model.fit(X_train, y_train, batch_size=32, epochs=200, validation_data=(X_test, y_test), callbacks=[cp_callback, lr_callback], shuffle=True)
    predicted_y = model.predict(X, verbose=1)
    matrix = confusion_matrix(y.argmax(axis=1), predicted_y.argmax(axis=1))
    print("Confusion matrix")
    print(matrix)

if __name__ == '__main__':
    main()
