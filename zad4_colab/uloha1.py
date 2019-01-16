import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import utils

train_img = pickle.load(open("gdrive/My Drive/dataset/train_img.p", "rb"))
train_labels = pickle.load(open("gdrive/My Drive/dataset/train_labels.p", "rb"))
valid_img = pickle.load(open("gdrive/My Drive/dataset/valid_img.p", "rb"))
valid_labels = pickle.load(open("gdrive/My Drive/dataset/valid_labels", "rb"))
test_img = pickle.load(open("gdrive/My Drive/dataset/test_img.p", "rb"))
test_labels = pickle.load(open("gdrive/My Drive/dataset/test_labels.p", "rb"))

neurons = 1024
act_func = "relu"
solver = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
input = (50*50*3,)
classes = 48
batch_size = 64

y_train = utils.to_categorical(train_labels, classes)
y_valid = utils.to_categorical(valid_labels, classes)
y_test = utils.to_categorical(test_labels, classes)

model = Sequential()
model.add(Dense(neurons, input_shape=input, activation=act_func))
model.add(Dense(classes, activation='softmax'))
model.summary()

model.compile(optimizer=solver,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_img, y_train,
                    batch_size=batch_size,
                    verbose=1,
                    epochs=20,
                    validation_data=(valid_img, y_valid))
score = model.evaluate(test_img, y_test, batch_size=128)
print('Test loss:', score[0])
print('Test accuracy:', score[1])