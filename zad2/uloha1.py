import os
import numpy as np
import pickle
import cv2


def euklid(image_array, name_of_dataset):
    if "Training" in name_of_dataset:
        max_images = 200
    else:
        max_images = 40

    for n in range(0, max_images):
        image = image_array[n]
        iterator = 1 + n
        while iterator < max_images:
            next_image = image_array[iterator]
            diff_image = image - next_image
            euklidean = cv2.norm(next_image, diff_image, 4)
            iterator += 1

    # return vzdialenost pixelov? alebo co to ma spravit
    # 100 * 100 a 3, tak porovnat kazdy pixel s hodnotou rgb, vzdialenost znamena rozdiel vo farbe jedneho pixelu
    # vstup budu nejake data bez ucitela, vystup priblizne triedy spravene pomocou clustering, moze byt 2-4-6 clustery.
    # budeme robit v centroidnych modeloch
    # 1 img porovnat so vsetkymi, 2 img porovnat so vsetkymi okrem prveho atd...
    # spravit sumu na variabli diff a to si ulozit a potom si povedat ze ked je euklidovska je napr 300k > tak to
    # znamena ze nepatri do classy


enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/Uloha2"
statistical_data = []

for clss in os.listdir(path):
    data = pickle.load(open(path+"/"+clss, "rb"))
    clss = clss.split()
    euklid(data, clss[1])
