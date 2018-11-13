import pickle
import matplotlib.pyplot as plt

clf = pickle.load(open("./model_clf_5000_n16", "rb"))
print(clf)
data = pickle.load(open("D:/skola/ZS 2018-2019/Suns_pycharm/Suns1/zad1/dataset test", "rb"))
labels = data["labels test"]
data = data["dataset test"]
test = []
for image in data:
    test.append(image.reshape(100 * 100 * 3))
for x in range(0, 2):
    skuska = clf.predict(test)
    print(skuska)
    n = 0
    chyba = 0
    for label in labels:
        if label != skuska[n]:
            chyba += 1
        n += 1
    print("Z 1920 testovacich vzoriek bolo zle zaklasifikovanych: " + str(chyba)
          + ", v percentach: " + str(chyba/19.2) + ".")
    plt.subplot()
