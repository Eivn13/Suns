import pickle
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt


def get_image(path):
    array = pickle.load(open(path, "rb"))
    index = random_number(len(array))
    return array[index]


def random_number(size):
    return random.randint(0, size-1)


end_dir = os.path.dirname(__file__)
end_dir = end_dir+"/Uloha2"
image_array = np.ndarray(shape=(4, 100, 100, 3), dtype=np.float32)
for parent in os.listdir(end_dir):
    if ("Test" in parent) or ("Training" in parent):
        for x in range(0, 4):
            image = get_image(end_dir+"/"+parent)
            image_array[x, :, :] = image

        img0 = cv2.cvtColor(image_array[0], cv2.COLOR_RGB2BGR)
        img1 = cv2.cvtColor(image_array[1], cv2.COLOR_RGB2BGR)
        img2 = cv2.cvtColor(image_array[2], cv2.COLOR_RGB2BGR)
        img3 = cv2.cvtColor(image_array[3], cv2.COLOR_RGB2BGR)

        plt.subplot(221), plt.imshow(img0), plt.title(parent)
        plt.subplot(222), plt.imshow(img1), plt.title(parent)
        plt.subplot(223), plt.imshow(img2), plt.title(parent)
        plt.subplot(224), plt.imshow(img3), plt.title(parent)

        plt.tight_layout()
        plt.imshow(img0)
        plt.waitforbuttonpress(15)
        plt.close()
