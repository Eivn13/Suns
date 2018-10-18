import os
import numpy as np
import pickle


def load_classes(path):
    all_classes_training = np.ndarray(shape=(200, 100, 100, 3), dtype=np.float32)
    all_classes_test = np.ndarray(shape=(40, 100, 100, 3), dtype=np.float32)
    for clss in os.listdir(path):
        if "Training" in clss:
            clss = pickle.load(open(path+"/"+clss, "rb"))
            all_classes_training = np.concatenate((all_classes_training, clss), axis=0)
        else:
            clss = pickle.load(open(path+"/"+clss, "rb"))
            all_classes_test = np.concatenate((all_classes_test, clss), axis=0)
    return all_classes_training, all_classes_test


enddir = os.path.dirname(__file__)
enddir = enddir[:-4]
path = "zad1/Uloha2"
dataset_training, dataset_test = load_classes(enddir+path)

