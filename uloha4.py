import os

# check if size of training files are the same
size_training = 0
size_test = 0
for file in os.listdir("Uloha2"):
    if ".py" not in file:
        pickle_file = file
        if "Training" in pickle_file:
            size_of_file = os.path.getsize("Uloha2/"+pickle_file)
            if size_training == 0:
                size_training = size_of_file
            if size_of_file != size_training:
                print("Wrong size of pickle file " + pickle_file + " dataset might be different")
        else:
            size_of_file = os.path.getsize("Uloha2/"+pickle_file)
            if size_test == 0:
                size_test = size_of_file
            if size_of_file != size_test:
                print("Wrong size of pickle file " + pickle_file + " dataset might be different")
