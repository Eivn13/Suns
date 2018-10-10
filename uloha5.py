import os
import numpy
import pickle


def randomize_two_arrays(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)
    return a, b


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
        dataset_test.extend(array)
        fruitname = [fruitname] * 40
        labels_test.extend(fruitname)
    if datatype == "Training":
        data_train = array[:160]
        data_valid = array[160:]
        dataset_training.extend(data_train)
        f = fruitname
        fruitname = [f] * 160
        labels_training.extend(fruitname)
        dataset_validation.extend(data_valid)
        fruitname_valid = [f] * 40
        labels_valid.extend(fruitname_valid)

# shuffle
dataset_training, labels_training = randomize_two_arrays(dataset_training, labels_training)
dataset_validation_valid, labels_valid = randomize_two_arrays(dataset_validation, labels_valid)
dataset_test, labels_test = randomize_two_arrays(dataset_test, labels_test)

save_training = {
    "dataset training": dataset_training,
    "labels training": labels_training
}

save_validation = {
    "dataset validation": dataset_validation,
    "labels validation": labels_valid,
}

save_test = {
    "dataset test": dataset_test,
    "labels test": labels_test
}

pickle.dump(save_training, open(os.path.dirname(__file__)+"/dataset training", "wb"))
pickle.dump(save_validation, open(os.path.dirname(__file__)+"/dataset validation", "wb"))
pickle.dump(save_test, open(os.path.dirname(__file__)+"/dataset test", "wb"))

