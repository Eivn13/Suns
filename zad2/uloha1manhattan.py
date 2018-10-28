from scipy.spatial import distance
import os
import pickle


def manhattan(i, j):
    return distance.cityblock(i, j)


enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/Uloha2"
statistical_data = []

for clss in os.listdir(path):
    data = pickle.load(open(path+"/"+clss, "rb"))
    statistical_data.append(clss)
    clss = clss.split()
    euklid(data, clss[1])
