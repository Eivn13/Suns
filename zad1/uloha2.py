import pickle
import cv2
import os
import numpy as np


def load_fruits(folder, fruit):
    fruitname = fruit[0]
    datatype = fruit[1]
    same_fruit = False
    global fruit_before
    if fruit_before == fruitname:   # check whether its the same fruit as before
        same_fruit = True
    if fruit_before == '' or fruit_before != fruitname:
        fruit_before = fruitname
    if datatype == "Training":
        number = subclasses_count_training.get(fruitname)
        tmp = num_of_images_training % number
        if tmp > 0 and same_fruit is False:     # if we load fruit for the first time add any remainder from modulo op.
            min_num_images = num_of_images_training / subclasses_count_training.get(fruitname)
            min_num_images = min_num_images + tmp
        else:
            min_num_images = num_of_images_training/subclasses_count_training.get(fruitname)
    else:
        number = subclasses_count_test.get(fruitname)
        tmp = num_of_images_test % number
        if tmp > 0 and same_fruit is False:
            min_num_images = num_of_images_test / subclasses_count_test.get(fruitname)
            min_num_images = min_num_images + tmp
        else:
            min_num_images = num_of_images_test/subclasses_count_test.get(fruitname)
    min_num_images = int(min_num_images)    # change float to int
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(min_num_images, image_size, image_size, ch), dtype=np.float32)

    # print(folder)
    num_images = 0
    for image in image_files:
        if num_images is min_num_images:
            break
        image_file = os.path.join(folder, image)
        try:
            img = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #  tak bude potrebne
            image_data = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if image_data.shape != (image_size, image_size, ch):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except (IOError, ValueError) as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def pickle_me_timbers(newfile, dataset):
    fruit = newfile[0]
    datatype = newfile[1]
    filepath = enddir+"/Uloha2/"+fruit+" "+datatype
    if os.path.exists(filepath):
        old_dataset = pickle.load(open(filepath, "rb"))
        dataset = np.concatenate((dataset, old_dataset), axis=0)
    pickle.dump(dataset, open(filepath, "wb"))


def count_subclasses(path):
    if len(subclasses_count_test) is 0:
        # count logic here
        for folder in os.listdir(path):
            name = folder
            name = name.split()
            key = name[0]
            if subclasses_count_test.get(key) is None:
                subclasses_count_test[key] = 1
            else:
                subclasses_count_test[key] += 1
    else:
        print("Subclasses already counted")


num_of_images_training = 200    # ~492, z tohto potom vyberieme 20/40 obrazkov na validaciu
num_of_images_test = 40   # ~162
image_size = 100
ch = 3
fruit_before = ''
subclasses_count_test = {}
subclasses_count_training = {}
enddir = os.path.dirname(__file__)
# dirname = enddir[:-6]
filename = os.path.join(enddir, "Fruits")
for parent in os.listdir(filename):
    if ("Test" in parent) or ("Training" in parent):
        count_subclasses(filename+"/"+parent)
        subclasses_count_training = subclasses_count_test
        for foldername in os.listdir(filename+"/"+parent):
            newfile = foldername
            newfile = newfile.split()
            try:
                newfile[1] = parent
            except IndexError:
                array = [newfile[0], parent]
                newfile = array
            # check if test or training, get num_of_images accordingly
            dataset = load_fruits(filename+"/"+parent+"/"+foldername, newfile)
            pickle_me_timbers(newfile, dataset)

# vstack
# hstack
# treba spravit,ze ked je 5 druhov jablk a chceme 200 obrazkov z jedneho naddruhu tak 200/5 pre kazde jabko - done
# potom z treningu vybrat obrazky v pomere 80/20 trening valid a test nechat tak
# vysledok maju byt 3 pickle subory
