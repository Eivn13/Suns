import os
import numpy as np
import pickle
import cv2


def euklid(image_array, name_of_dataset):
    if "Training" in name_of_dataset:
        max_images = 200
    else:
        max_images = 40
    
    avg_img = np.ndarray(shape=(100, 100, 3), dtype=np.float32)
    array = []
    avg_array = []
    
    for image in image_array:
        avg_img = avg_img + image
    
    avg_img = avg_img/max_images

    for n in range(0, max_images):  
        image = image_array[n]
        avg_array.append(cv2.norm(next_image, avg_image, 4))    # C
        iterator = 1 + n
        while iterator < max_images:
            next_image = image_array[iterator]
            diff_image = image - next_image
            # asi bug, preco porovnavam next_image a odpocitane img a next image?
            euklidean = cv2.norm(next_image, diff_image, 4) # porovnaj vzdialenost od dalsieho obrazku
            array.append(euklidean)
            iterator += 1
            
    global statistical_data
    statistical_data.append(array)
    statistical_data.append("C")
    statistical_data.append(avg_array)

    # return vzdialenost pixelov? alebo co to ma spravit
    # 100 * 100 a 3, tak porovnat kazdy pixel s hodnotou rgb, vzdialenost znamena rozdiel vo farbe jedneho pixelu
    # vstup budu nejake data bez ucitela, vystup priblizne triedy spravene pomocou clustering, moze byt 2-4-6 clustery.
    # budeme robit v centroidnych modeloch
    # 1 img porovnat so vsetkymi, 2 img porovnat so vsetkymi okrem prveho atd...
    # spravit sumu na variable diff a to si ulozit a potom si povedat ze ked je euklidovska je napr 300k > tak to
    # znamena ze nepatri do classy
    # cv2.resize na zmensenie obrazku 4x napr
    # np.linalg.norm


enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/Uloha2"
statistical_data = []

for clss in os.listdir(path):
    data = pickle.load(open(path+"/"+clss, "rb"))
    statistical_data.append(clss)
    clss = clss.split()
    euklid(data, clss[1])
