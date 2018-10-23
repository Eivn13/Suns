from sklearn.cluster import KMeans
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

enddir = os.path.dirname(__file__)[:-4]
path = enddir+"zad1/Uloha2"

image_array = pickle.load((open(path+"/Apple Test", "rb")))
x, y, z = image_array[10].shape
img = image_array[10].reshape(x*y, z)
kmeans_cluster = KMeans(n_clusters=3)
kmeans_cluster.fit(img)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_
plt.figure(figsize=(15, 8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))
plt.waitforbuttonpress(30)
plt.close()
