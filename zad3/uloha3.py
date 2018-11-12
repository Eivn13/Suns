import os
import pickle
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
import numpy as np


def mlp(dataset, size):
    clf = MLPClassifier(solver="adam",
                        hidden_layer_sizes=(25,),
                        max_iter=5000,
                        learning_rate_init=0.0003,
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
        if size < n:
            break
        X.append(x[0])
        y.append(x[1])
        n += 1
    model = clf.fit(X, y)
    plot_loss = np.asarray(model.loss_curve_)
    plt.plot(plot_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    global test
    data = []
    labels = []
    for x in test:
        data.append(x[0])
        labels.append(x[1])
    skuska = clf.predict(data)
    print(skuska)
    n = 0
    chyba = 0
    for label in labels:
        if label != skuska[n]:
            chyba += 1
        n += 1
    print(  # "Velkost naucneho datasetu: " + str(size) + ", " +
        "pocet chyb: " + str(chyba))
    plt.show()
    pickle.dump(clf, open("./model_clf", "wb"))


def support_vector_machine(dataset, size):
    print("hello")


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
    mlp(dataset, 5000)
    # support_vector_machine(dataset, 5)


dataset = []
test = []
compute()
