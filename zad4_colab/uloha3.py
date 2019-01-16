import pickle
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import utils
from keras import regularizers
from keras import callbacks

train_img = pickle.load(open("gdrive/My Drive/dataset/train_img.p", "rb"))
train_labels = pickle.load(open("gdrive/My Drive/dataset/train_labels.p", "rb"))
valid_img = pickle.load(open("gdrive/My Drive/dataset/valid_img.p", "rb"))
valid_labels = pickle.load(open("gdrive/My Drive/dataset/valid_labels", "rb"))
test_img = pickle.load(open("gdrive/My Drive/dataset/test_img.p", "rb"))
test_labels = pickle.load(open("gdrive/My Drive/dataset/test_labels.p", "rb"))


def neuronka(rate):
    neurons = 1024
    act_func = "relu"
    solver = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    input = (50 * 50 * 3,)
    classes = 48
    batch_size = 64

    n = 0
    X = []
    for img in train_img:
        if n < rate:
            X.append(img)
        n += 1
    n = 0
    y = []
    for label in train_labels:
        if n < rate:
            y.append(label)
        n += 1

    X = np.asarray(X)
    y = np.asarray(y)
    y_train = utils.to_categorical(y, classes)
    y_valid = utils.to_categorical(valid_labels, classes)
    y_test = utils.to_categorical(test_labels, classes)

    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

    model = Sequential()
    model.add(Dense(neurons, input_shape=input, activation=act_func, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(classes, activation='softmax'))

    model.compile(optimizer=solver,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X, y_train,
                        batch_size=batch_size,
                        verbose=1,
                        epochs=20,
                        callbacks=[early_stop],
                        validation_data=(valid_img, y_valid))

    score = model.evaluate(test_img, y_test, batch_size=128, verbose=0)
    global train_acc, valid_acc, test_acc
    train_acc.append(history.history['acc'][-1])
    valid_acc.append(history.history['val_acc'][-1])
    test_acc.append(score[1])


rates = [100, 500, 1000, 3000, 5000, 8563]
train_acc = []
valid_acc = []
test_acc = []
for num in rates:
    neuronka(num)

plt.title("Effect of batch size on accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Batch size")
plt.plot(rates, train_acc)
plt.plot(rates, valid_acc)
plt.plot(rates, test_acc)
plt.legend(['training accuracy', 'validation accuracy', 'testing accuracy'])
plt.show()
