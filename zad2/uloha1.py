import os
import numpy as np
import pickle


def manhattan(image_array):
    print("placeholder")
#     return vzdialenost pixelov? alebo co to ma spravit


enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/Uloha2"
statistical_data = []

for clss in os.listdir(path):
    data = pickle.load(open(path+"/"+clss, "rb"))
    manhattan(data)
