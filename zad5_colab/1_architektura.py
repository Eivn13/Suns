import pickle
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras import utils
from keras import regularizers
from keras import callbacks

# moja neuronka

train = pickle.load(open("gdrive/My Drive/dataset/train50x50.pickle", "rb"))
valid = pickle.load(open("gdrive/My Drive/dataset/valid50x50.pickle", "rb"))
test = pickle.load(open("gdrive/My Drive/dataset/test50x50.pickle", "rb"))
train_img = []
train_labels = []
valid_img = []
valid_labels = []
test_img = []
test_labels = []
for data in train:
    train_img.append(data[0])
    train_labels.append(data[1])
for data in valid:
    valid_img.append(data[0])
    valid_labels.append(data[1])
for data in test:
    test_img.append(data[0])
    test_labels.append(data[1])

# change to nparray
train_img = np.array(train_img)
train_labels = np.array(train_labels)
valid_img = np.array(valid_img)
valid_labels = np.array(valid_labels)
test_img = np.array(test_img)
test_labels = np.array(test_labels)


def neuronka():
    neurons = 200
    act_func = "relu"
    solver = Adam(lr=0.1, epsilon=0.1)
    classes = 48
    batch_size = 64
    epochs = 20

    y_train = utils.to_categorical(train_labels, classes)
    y_valid = utils.to_categorical(valid_labels, classes)
    y_test = utils.to_categorical(test_labels, classes)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')

    # layers
    model = Sequential()
    model.add(Conv2D(50, kernel_size=3, activation='relu', input_shape=(50, 50, 3)))
    model.add(MaxPooling2D(pool_size=(10, 10), padding='valid', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(neurons, activation=act_func, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))
    model.summary()

    model.compile(optimizer=solver,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_img, y_train,
                        batch_size=batch_size,
                        verbose=0,
                        epochs=epochs,
                        callbacks=[early_stop],
                        validation_data=(valid_img, y_valid))

    score = model.evaluate(test_img, y_test, batch_size=128, verbose=1)
    print(score)
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


neuronka()
