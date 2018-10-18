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
end_dir = end_dir+"/Fruits"
image_array = np.ndarray(shape=(4, 100, 100, 3), dtype=np.float32)
for parent in os.listdir(end_dir):
    if ("Test" in parent) or ("Training" in parent):
        for fruit_folder in os.listdir(end_dir + "/" + parent):
            x = 0
            for img in os.listdir(end_dir + "/" + parent + "/" + fruit_folder):
                image = cv2.imread(end_dir + "/" + parent + "/" + fruit_folder + "/" + img)
                image_data = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                image_array[x, :, :] = image_data
                x += 1
                if x > 3:
                    break

            img0 = cv2.cvtColor(image_array[0], cv2.COLOR_RGB2BGR)
            img1 = cv2.cvtColor(image_array[1], cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(image_array[2], cv2.COLOR_RGB2BGR)
            img3 = cv2.cvtColor(image_array[3], cv2.COLOR_RGB2BGR)

            plt.subplot(221), plt.imshow(img0), plt.title(fruit_folder + " " + parent)
            plt.subplot(222), plt.imshow(img1), plt.title(fruit_folder + " " + parent)
            plt.subplot(223), plt.imshow(img2), plt.title(fruit_folder + " " + parent)
            plt.subplot(224), plt.imshow(img3), plt.title(fruit_folder + " " + parent)

            plt.tight_layout()
            plt.imshow(img0)
            plt.waitforbuttonpress(5)
            plt.close()
