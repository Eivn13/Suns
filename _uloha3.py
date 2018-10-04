import pickle
import cv2
import os
import random


def showImage(path):
    array = pickle.load(open(path, "rb"))     # vytiahni z pickle obrazky
    index = randomNumber(len(array))    # Potom ukaz obrazok na indexe, ktory bude random vybraty z randomnumber funkcie
    cv2.imshow("window", array[index])
    print(array[index])
    cv2.waitKey(0)


def randomNumber(size):
    return random.randint(0, size-1)


enddir = os.path.dirname(__file__)
enddir = enddir+"/Uloha2"
for parent in os.listdir(enddir):
    if ("Test" in parent) or ("Training" in parent):
        for x in range(0, 4):
            showImage(enddir+"/"+parent)
