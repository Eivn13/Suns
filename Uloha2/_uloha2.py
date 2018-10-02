import pickle
import cv2
import os
import numpy as np
import imageio


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size, 3),
                         dtype=np.float32)
    print(folder)
    print(dataset[0].shape)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(folder, image)
        try:
            image_data = (imageio.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size, 3):
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


def load_fruits(path, filename, newfile):
    images = []
    for file in os.listdir(path):
        if (".jpg" in file) or (".png" in file) or (".jpeg" in file):   # ako pozriet ci je image corrupted?
            imgname = file    # vloz meno
            img = cv2.imread(path+"/"+imgname)
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            normalizedimg = cv2.normalize(grayimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(normalizedimg)
    # skonci sa for, mame vsetky obrazky znormalizovane
    print(filename)
    pickle.dump(images, open(enddir+"/"+newfile[0]+" "+newfile[1], "wb"))


enddir = os.path.dirname(__file__)
dirname = enddir[:-6]
filename = os.path.join(dirname, "Fruits/fruits/fruits-360")
image_size = 100
pixel_depth = 3

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
            # load_fruits(filename+"/"+parent+"/"+foldername, nameoffile, newfile)
            load_letter(filename+"/"+parent+"/"+foldername, 100)
