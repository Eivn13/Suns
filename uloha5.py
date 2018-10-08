import os
import numpy as np
import pickle


# nacitaj data z picklov

# rozdel trening na training a validation 160-40

# zakoduj triedy na indexy, potom poprehadzuj indexy?

enddir = os.path.dirname(__file__)
enddir = enddir+"/Uloha2"
dataset_training = np.ndarray(shape=(48 * 160, 100, 100, 3), dtype=np.float32)
ele_training = 0
dataset_validation = np.ndarray(shape=(48 * 40, 100, 100, 3), dtype=np.float32)
dataset_test = np.ndarray(shape=(48 * 40, 100, 100, 3), dtype=np.float32)
ele_test = 0
for fruit in os.listdir(enddir):
    array = pickle.load(open(enddir+"/"+fruit, "rb"))
    fruit = fruit.split()
    fruitname = fruit[0]
    datatype = fruit[1]
    for x in array:
        if datatype == "Test":
            if ele_test == 40:
                ele_test = 0
            dataset_test[ele_test] = x
            ele_test += 1
        if datatype == "Training":
            if ele_training == 200:
                ele_training = 0
            if ele_training == 160:
                dataset_validation[ele_training-160] = x
                ele_training += 1
            else:
                dataset_training[ele_training] = x
                ele_training += 1
    ele_test = 0
    ele_training = 0
pickle.dump(dataset_training, open(os.path.dirname(__file__)+"/dataset_training", "wb"))
pickle.dump(dataset_validation, open(os.path.dirname(__file__)+"/dataset_validation", "wb"))
pickle.dump(dataset_test, open(os.path.dirname(__file__)+"/dataset_test", "wb"))
