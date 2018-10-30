import os
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt


def plot(data):
    n = 0
    euk_max = 0
    euk_max_index = ''
    euk_min = 50000000000
    euk_min_index = ''
    man_max = 0
    man_max_index = ''
    man_min = 50000000000
    man_min_index = ''
    avg_min = 50000000000
    avg_min_index = ''
    avg_max = 0
    avg_max_index = ''
    names = []
    euk_plot = []
    man_plot = []
    avg_plot = []
    while n < len(data):
        euk = sum(data[n+1])/len(data[n+1])
        man = sum(data[n+2])/len(data[n+2])
        avg = sum(data[n+3])/len(data[n+3])
        if euk < euk_min:
            euk_min = euk
            euk_min_index = n
        if euk > euk_max:
            euk_max = euk
            euk_max_index = n
        if man < man_min:
            man_min = man
            man_min_index = n
        if man > man_max:
            man_max = man
            man_max_index = n
        if avg < avg_min:
            avg_min = avg
            avg_min_index = n
        if avg > avg_max:
            avg_max = avg
            avg_max_index = n
        names.append(data[n])
        euk_plot.append(sum(data[n+1]))
        man_plot.append(sum(data[n+2]))
        avg_plot.append(sum(data[n+3]))
        print(data[n], euk_plot[-1], man_plot[-1], avg_plot[-1])
        n += 4

    print("Najrozmanitejsia trieda podla euklidovho algoritmu je " + data[euk_max_index] +
          ". Najmenej rozmanita trieda je " + data[euk_min_index] + ".")
    print("Najrozmanitejsia trieda podla manhattan algoritmu je " + data[man_max_index] +
          ". Najmenej rozmanita trieda je " + data[man_min_index] + ".")
    print("Trieda s najvacsou odchylkou od priemernych dat je " + data[avg_max_index] +
          ". Trieda s najmensou odchylkou od priemernych dat je " + data[avg_min_index] + ".")

    plt.hist(names, weights=euk_plot, bins=np.arange(50) - 0.5, edgecolor='k')
    plt.title("Suma vzdialenosti tried podla Euklidovho algoritmu")
    plt.xlabel("Ovocie")
    plt.ylabel("Vzdialenost")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.hist(names, weights=man_plot, bins=np.arange(50)-0.5, edgecolor='k')
    plt.title("Suma vzdialenosti tried podla Manhattanskeho algoritmu")
    plt.xlabel("Ovocie")
    plt.ylabel("Vzdialenost")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def compute(image_array, name_of_class, name_of_dataset):
    if "Training" in name_of_dataset:
        max_images = 200
    else:
        max_images = 40

    avg_img = np.zeros(shape=(50, 50, 3), dtype=np.float32)
    array1 = []
    array2 = []
    avg_array = []

    for image in image_array:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        avg_img = avg_img + image

    avg_img = avg_img / max_images

    n = 0
    for image in image_array:
        if n >= max_images:
            break
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        avg_array.append(cv2.norm(image, avg_img, 4))  # C
        iterator = 1 + n
        while iterator < max_images:
            next_image = image_array[iterator]
            next_image = cv2.resize(next_image, (0, 0), fx=0.5, fy=0.5)
            euklidean = cv2.norm(image, next_image, 4)  # porovnaj vzdialenost od dalsieho obrazku, 4 = L2
            array1.append(euklidean)
            manhattan = cv2.norm(image, next_image, 2)  # 2 = l1
            array2.append(manhattan)
            iterator += 1
        n = n + 1

    number = 0
    global statistical_data
    global done_classes
    if name_of_class in done_classes:
        number = done_classes.index(name_of_class)
        if number > 0:
            number *= 4
        statistical_data[number+1] += array1
        statistical_data[number+2] += array2
        statistical_data[number+3] += avg_array
    else:
        done_classes.append(name_of_class)
        statistical_data.append(name_of_class)
        statistical_data.append(array1)
        statistical_data.append(array2)
        statistical_data.append(avg_array)


enddir = os.path.dirname(__file__)[:-4]
path = enddir + "zad1/Uloha2"
statistical_data = []
done_classes = []

for clss in os.listdir(path):
    data = pickle.load(open(path+"/"+clss, "rb"))
    clss = clss.split()
    compute(data, clss[0], clss[1])
plot(statistical_data)
