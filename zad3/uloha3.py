import os
import pickle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import random
import matplotlib.pyplot as plt
import numpy as np


def mlp(dataset, size):
    clf = MLPClassifier(solver="adam",
                        hidden_layer_sizes=(1, 5), max_iter=1000, learning_rate_init=0.003, learning_rate="invscaling")
    X = []
    y = []
    n = 1
    for x in dataset:
        if size < n:
            break
        X.append(x[0])
        y.append(x[1])
        n += 1
    print(y)
    model = clf.fit(X, y)
    plot_loss = np.asarray(model.loss_curve_)
    plt.plot(plot_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    global test
    skuska = clf.predict(test)
    print(skuska)
    print("Loss: 1, " + str(plot_loss[0]) + " 2, " + str(plot_loss[1]) + " 3, " + str(plot_loss[2]) + " ..." +
          str(len(plot_loss)) + "-th, " + str(plot_loss[-1]))
    # plt.plot(skuska.)


def support_vector_machine(dataset, size):
    print("hello")


def make_dataset():
    size = 5000
    size_of_sample = 112
    enddir = os.path.dirname(__file__)[:-4]
    path = enddir + "zad1/Uloha2"
    statistical_data = []
    done_classes = []

    for clss in os.listdir(path):
        data = pickle.load(open(path + "/" + clss, "rb"))
        clss = clss.split()
        if "Training" in clss[1]:
            global dataset
            n = 0
            for image in data:
                if n == size_of_sample:
                    break
                dataset.append(tuple((image.reshape(100*100*3), clss[0])))
                n += 1
            size_of_sample = 104
        else:
            global test
            for image in data:
                test.append(image.reshape(100*100*3))

        random.shuffle(dataset)


def compute():
    make_dataset()
    mlp(dataset, 5)
    # support_vector_machine(dataset)

    " tu sa bude volat svm 1 svm 2 mlp 1 mlp 2"


dataset = []
test = []
compute()
