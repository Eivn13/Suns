import pickle
import cv2
import os
import numpy as np
import imageio


def load_fruit(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size, ch),
                         dtype=np.float32)
    print(folder)
    print(dataset[0].shape)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
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

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def pickle_me_timbers(newfile, dataset):
    pickle.dump(dataset, open(enddir+"/"+newfile[0]+" "+newfile[1], "wb"))


enddir = os.path.dirname(__file__)
dirname = enddir[:-6]
filename = os.path.join(dirname, "Fruits")
image_size = 100
pixel_depth = 255
ch = 3

for parent in os.listdir(filename):
    if ("Test" in parent) or ("Training" in parent):
        for foldername in os.listdir(filename+"/"+parent):
            nameoffile = foldername+" "+parent+".p"
            newfile = foldername
            newfile = newfile.split()
            try:
                newfile[1] = parent
            except IndexError:
                array = [newfile[0], parent]
                newfile = array
            dataset = load_fruit(filename+"/"+parent+"/"+foldername, 100)
            pickle_me_timbers(newfile, dataset)
