import os
import pickle
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
import numpy as np


def mlp(dataset, size, hidden_layer):
    clf = MLPClassifier(solver="adam",
                        hidden_layer_sizes=hidden_layer,
                        max_iter=5000,
                        learning_rate_init=0.003,
                        learning_rate="adaptive",
                        activation='relu',
                        alpha=0.1,
                        n_iter_no_change=10,
                        tol=0.000001,
                        verbose=1)
    X = []
    y = []
    n = 0
    for x in dataset:
        if size == n:
            break
        X.append(x[0])
        y.append(x[1])
        n += 1
    clf.fit(X, y)
    global test
    data = []
    labels = []
    for x in test:
        data.append(x[0])
        labels.append(x[1])
    skuska = clf.predict(data)
    n = 0
    chyba = 0
    for label in labels:
        if label != skuska[n]:
            chyba += 1
        n += 1
    print("Velkost naucneho datasetu: " + str(size) + ", " + "accuracy: " + str(100 - (chyba/19.2)))
    return 100 - (chyba/19.2)


def support_vector_machine(dataset, size, krnl):
    X = []
    y = []
    n = 0

    for j in dataset:
        if size == n:
            break
        X.append(j[0])
        y.append(j[1])
        n += 1

    global test
    data = []
    labels = []
    for x in test:
        data.append(x[0])
        labels.append(x[1])

    clf = svm.SVC(kernel=krnl, cache_size=4000, gamma="scale")
    clf.fit(X, y)
    skuska = clf.predict(data)
    n = 0
    chyba = 0
    for label in labels:
        if label != skuska[n]:
            chyba += 1
        n += 1

    print("Velkost naucneho datasetu: " + str(size) + ", " + "accuracy: " + str(100 - (chyba/19.2)))
    return 100 - (chyba/19.2)


def make_dataset():
    size_of_sample = 112
    enddir = os.path.dirname(__file__)[:-4]
    path = enddir + "zad1/Uloha2"

    for clss in os.listdir(path):
        data = pickle.load(open(path + "/" + clss, "rb"))
        clss = clss.split()
        if "Training" in clss[1]:
            global dataset
            n = 0
            for image in data:
                if n == size_of_sample:
                    break
                dataset.append(tuple((image.reshape(100 * 100 * 3), clss[0])))
                n += 1
            size_of_sample = 104
        else:
            global test
            for image in data:
                test.append(tuple((image.reshape(100 * 100 * 3), clss[0])))

    random.shuffle(dataset)


def compute():
    make_dataset()
    one_hidden = tuple((100, ))
    n_hidden = tuple((50, 40, 30))
    ds = [50, 100, 200, 1000, 5000]
    array = [mlp(dataset, ds[0], one_hidden), mlp(dataset, ds[1], one_hidden), mlp(dataset, ds[2], one_hidden),
             mlp(dataset, ds[3], one_hidden), mlp(dataset, ds[4], one_hidden)]
    plt.ylabel("Accuracy")
    plt.xlabel("Size of training dataset")
    plt.plot([50, 100, 200, 1000, 5000], array)

    del array[:]
    array = [mlp(dataset, ds[0], n_hidden), mlp(dataset, ds[1], n_hidden), mlp(dataset, ds[2], n_hidden),
             mlp(dataset, ds[3], n_hidden), mlp(dataset, ds[4], n_hidden)]
    plt.plot([50, 100, 200, 1000, 5000], array)

    del array[:]
    array = [support_vector_machine(dataset, ds[0], "linear"), support_vector_machine(dataset, ds[1], "linear"),
             support_vector_machine(dataset, ds[2], "linear"), support_vector_machine(dataset, ds[3], "linear"),
             support_vector_machine(dataset, ds[4], "linear")]
    plt.plot([50, 100, 200, 1000, 5000], array)

    del array[:]
    array = [support_vector_machine(dataset, ds[0], "rbf"), support_vector_machine(dataset, ds[1], "rbf"),
             support_vector_machine(dataset, ds[2], "rbf"), support_vector_machine(dataset, ds[3], "rbf"),
             support_vector_machine(dataset, ds[4], "rbf")]
    plt.plot([50, 100, 200, 1000, 5000], array)

    plt.legend(["1 hidden layer with 100 neurons", "3 hidden layers with 50, 40, 30 neurons",
                "SVM linear", "SVM rbf"])
    plt.show()


dataset = []
test = []
compute()
