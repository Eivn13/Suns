import os
import cv2
import numpy as np
#curpath = "./Fruits/fruits/fruits-360/Test/"
#curpath = "./Fruits/fruits/fruits-360/Training/"


def zrut(path):
    for filename in os.listdir(path):
        for image in os.listdir(path+filename):
            #print(image) # nasli sme meno fileu
            try:
                #print(path+filename+"/"+image)
                img = cv2.imread(path + filename + "/" + image, 0)
                cv2.imshow('window', img)
                cv2.waitKey(0)
                try:
                    ayy = cv2.imdecode(img, 0) #preco returnuje none?
                    if ayy is None:
                        print("Corrupted image: ", path + filename + "/" + image)
                except (BufferError, AssertionError) as e:
                    print("ayy")#path + filename + image)
            except(IOError, SyntaxError) as e:
                print("Bad image:", image)


zrut("./Fruits/fruits/fruits-360/Test/")
zrut("./Fruits/fruits/fruits-360/Training/")
