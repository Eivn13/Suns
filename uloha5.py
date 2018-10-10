import os
import numpy as np
import pickle
from random import shuffle

# nacitaj data z picklov

# rozdel trening na training a validation 160-40

# zakoduj triedy na indexy, potom poprehadzuj indexy?


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


enddir = os.path.dirname(__file__)
enddir = enddir+"/Uloha2"
dataset_training = []
labels_training = []
dataset_validation = []
labels_test = []
dataset_test = []
labels_valid = []


for fruit in os.listdir(enddir):
    array = pickle.load(open(enddir+"/"+fruit, "rb"))
    fruit = fruit.split()
    fruitname = fruit[0]
    datatype = fruit[1]
    if datatype == "Test":
        dataset_test.append(array)
        labels_test.extend(fruitname)
    if datatype == "Training":
        data_train = array[:160]
        data_valid = array[160:]
        dataset_training.append(data_train)
        labels_training.extend(fruitname)
        dataset_validation.append(data_valid)
        labels_valid.extend(fruitname)

# shuffle
train_dataset, labels_training = randomize(dataset_training, labels_training)
test_dataset, labels_test = randomize(dataset_test, labels_test)
valid_dataset, labels_valid = randomize(dataset_validation, labels_valid)

# save
save_training = {
    "dataset_training": train_dataset,
    "labels_training": labels_training,
}
save_test = {
    "dataset_test": test_dataset,
    "labels_test": labels_test,
}
save_validation = {
    "dataset_validation": valid_dataset,
    "labels_valid": labels_valid,
}

pickle.dump(save_training, open(os.path.dirname(__file__)+"/dataset_training", "wb"))
pickle.dump(save_validation, open(os.path.dirname(__file__)+"/dataset_validation", "wb"))
pickle.dump(save_test, open(os.path.dirname(__file__)+"/dataset_test", "wb"))

