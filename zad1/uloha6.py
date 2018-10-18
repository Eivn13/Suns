import pickle
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt

end_dir = os.path.dirname(__file__)
for parent in os.listdir(end_dir):
    if "dataset" in parent:
        print(end_dir+"/"+parent)
        dataset = pickle.load(open(end_dir+"/"+parent, "rb"))
        images = dataset[parent]
        parent = parent.split()
        label_name = parent[1]
        labels = dataset["labels "+label_name]
        for x in range(0, 4):
            img0 = cv2.cvtColor(images[0 + x], cv2.COLOR_RGB2BGR)
            img1 = cv2.cvtColor(images[1 + x], cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(images[2 + x], cv2.COLOR_RGB2BGR)
            img3 = cv2.cvtColor(images[3 + x], cv2.COLOR_RGB2BGR)

            plt.subplot(221), plt.imshow((img0 * 255).astype(np.uint8)), plt.title(labels[0 + x])
            plt.subplot(222), plt.imshow((img1 * 255).astype(np.uint8)), plt.title(labels[1 + x])
            plt.subplot(223), plt.imshow((img2 * 255).astype(np.uint8)), plt.title(labels[2 + x])
            plt.subplot(224), plt.imshow((img3 * 255).astype(np.uint8)), plt.title(labels[3 + x])

            plt.tight_layout()
            plt.waitforbuttonpress(30)
            plt.close()
