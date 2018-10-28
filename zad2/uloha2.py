from sklearn.cluster import KMeans
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


kmeans_cluster = KMeans(verbose=1)

print(dataset.shape)

kmeans_cluster.fit(dataset)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
image_cluster = cluster_centers.reshape(-1, 50, 50, 3)

n = 0
labels = [[] for i in range(8)]
for label in cluster_labels:
    labels[label].append(image_array["labels validation"][n])

    n += 1

n = 0
for label in labels:
    print(str(n) + " label contains these fruits: " + str(np.unique(label)))
    n += 1

n = 241
for img in image_cluster:
    tmp = img.astype(np.float32)
    img = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    plt.subplot(n), plt.imshow(img)
    if n < 249:
        n = n + 1

plt.tight_layout()
plt.waitforbuttonpress(300000)
plt.close()
