import os
import pickle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np


def mlp(dataset, size, hidden_layer):
    priem_chybovost = 0
    jump = int(round(size/5))
    for x in range(0, size, jump):
        clf = MLPClassifier(solver="adam",
                            hidden_layer_sizes=hidden_layer,
                            max_iter=5000,
                            learning_rate_init=0.003,
                            learning_rate="adaptive",
                            activation='relu',
                            alpha=0.1,
                            n_iter_no_change=10,
                            tol=0.000001)
        X = []
        y = []
        n = 0
        for j in dataset:
            if size == n:
                break
            X.append(j[0])
            y.append(j[1])
            n += 1
        data = []
        labels = []
        data = X[x:x+jump]
        labels = y[x:x+jump]
        del X[x:x+jump], y[x:x+jump]
        model = clf.fit(X, y)
        # plot_loss = np.asarray(model.loss_curve_)
        # plt.plot(plot_loss)
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.show()
        skuska = clf.predict(data)
        n = 0
        chyba = 0
        for label in labels:
            if label != skuska[n]:
                chyba += 1
            n += 1

        priem_chybovost += chyba/(jump/100)
        print("Velkost trenovacich dat: " + str(size-jump) + ", " + "pocet chyb: " + str(chyba))
        pickle.dump(clf, open("./model_clf_" + str(size) + "_" + str(hidden_layer), "wb"))

    return 100 - (priem_chybovost / 5)


def support_vector_machine(dataset, size, krnl):
    priem_chybovost = 0
    jump = int(round(size / 5))
    for x in range(0, size, jump):
        X = []
        y = []
        n = 0

        for j in dataset:
            if size == n:
                break
            X.append(j[0])
            y.append(j[1])
            n += 1
        data = []
        labels = []
        data = X[x:x + jump]
        labels = y[x:x + jump]
        del X[x:x + jump], y[x:x + jump]

        clf = svm.SVC(kernel=krnl, cache_size=4000, gamma="scale")
        clf.fit(X, y)
        skuska = clf.predict(data)
        n = 0
        chyba = 0
        for label in labels:
            if label != skuska[n]:
                chyba += 1
            n += 1

        priem_chybovost += chyba/(jump/100)
        print("Velkost trenovacich dat: " + str(size-jump) + ", " + "pocet chyb: " + str(chyba))
        pickle.dump(clf, open("./model_clf_svm_" + krnl, "wb"))

    return 100 - (priem_chybovost / 5)


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
                tmp = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                dataset.append(tuple((tmp.reshape(50 * 50 * 3), clss[0])))
                n += 1
            size_of_sample = 104

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
compute()
