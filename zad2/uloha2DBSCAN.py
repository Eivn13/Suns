from sklearn.cluster import DBSCAN
import pickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/dataset validation"
x, y, z = (50, 50, 3)
image_array = pickle.load((open(path, "rb")))
dataset = np.ndarray(shape=(len(image_array["dataset validation"]), 7500), dtype=np.float64)
n = 0
for image in image_array["dataset validation"]:
    tmp = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    dataset[n] = tmp.reshape(x*y*z)
    n = n + 1

clustering = DBSCAN(eps=10, min_samples=40, n_jobs=4).fit(dataset)
labels = np.unique(clustering.labels_)
print(labels-1)

n = 0
avg_images = [[] for i in range(len(labels))]
for image in dataset:
    avg_images[clustering.labels_[n]].append(image)
    n += 1
del avg_images[-1]

plt_images = []
n = 1
for avg_image in avg_images:
    if n > 16:
        break
    img = sum(avg_image)/len(avg_image)
    tmp = img.astype(np.float32)
    plt.subplot(4, 4, n), plt.imshow(cv2.cvtColor(tmp.reshape(50, 50, 3), cv2.COLOR_BGR2RGB)), plt.title(n-1)
    n += 1
plt.tight_layout()
plt.show()
