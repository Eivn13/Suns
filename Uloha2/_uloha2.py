import pickle
import cv2
import os


def picklemaker(path, filename):
    images = []
    for file in os.listdir(path):
        # predpokladajme ze path prisiel nakonci s /
        if (".jpg" in file) or (".png" in file) or (".jpeg" in file):   # ako pozriet ci je image corrupted?
            imgname = file    # vloz meno
            img = cv2.imread(path+"/"+imgname)
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            normalizedimg = cv2.normalize(grayimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(normalizedimg)
    # skonci sa for, mame vsetky obrazky znormalizovane
    print(filename)
    pickle.dump(images, open(enddir+"/"+filename, "wb"))


enddir = os.path.dirname(__file__)
dirname = enddir[:-6]
filename = os.path.join(dirname, "Fruits/fruits/fruits-360")
for parent in os.listdir(filename):
    if ("Test" in parent) or ("Training" in parent):
        for foldername in os.listdir(filename+"/"+parent):
            nameoffile = foldername+" "+parent+".p"
            picklemaker(filename+"/"+parent+"/"+foldername, nameoffile)


# TODO: image corrupted, pozriet ci je spravne dane do pola