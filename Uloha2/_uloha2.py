import pickle
import cv2
import os
import numpy as np


def load_fruits(folder, min_num_images):
    image_files = os.listdir(folder)
    diff = len(image_files) - 100
    image_files = image_files[:-diff]
    dataset = np.ndarray(shape=(100, image_size, image_size, ch),
                         dtype=np.float32)
    # print(folder)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            img = cv2.imread(image_file, cv2.IMREAD_COLOR)
            image_data = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if image_data.shape != (image_size, image_size, ch):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' %
                        (num_images, min_num_images))


def pickle_me_timbers(newfile, dataset):
    # print(enddir+"/"+newfile[0]+" "+newfile[1])
    if os.path.exists(enddir+"/"+newfile[0]+" "+newfile[1]):
        print("%s dataset exists - Skipping pickling." % newfile[0])
    else:
        pickle.dump(dataset, open(enddir+"/"+newfile[0]+" "+newfile[1], "wb"))


def count_subclasses(path):
    if len(subclasses_count) is 0:
        # count logic here
        for folder in os.listdir(path):
            name = folder
            name = name.split()
            key = name[0]
            if subclasses_count.get(key) is None:
                subclasses_count[key] = 1
            else:
                subclasses_count[key] += 1
        return subclasses_count
    else:
        print("Subclasses already counted")


num_of_images_test = 100
num_of_images_training = 100
num_of_images_validation = 100
image_size = 100
ch = 3
subclasses_count = {}
enddir = os.path.dirname(__file__)
dirname = enddir[:-6]
filename = os.path.join(dirname, "Fruits")
for parent in os.listdir(filename):
    if ("Test" in parent) or ("Training" in parent):
        count_subclasses(filename+"/"+parent)
        for foldername in os.listdir(filename+"/"+parent):
            newfile = foldername
            newfile = newfile.split()
            try:
                newfile[1] = parent
            except IndexError:
                array = [newfile[0], parent]
                newfile = array
            # check if test or training, get num_of_images accordingly
            dataset = load_fruits(filename+"/"+parent+"/"+foldername, 100)
            pickle_me_timbers(newfile, dataset)

# vstack
# hstack
# treba spravit,ze ked je 5 druhov jablk a chceme 200 obrazkov z jedneho nad druhu tak 200/5 pre kazde jabko
# potom z treningu vybrat obrazky v pomere 80/20 trening valid a test nechat tak
# vysledok maju byt 3 pickle subory
