

""" WARNING : EXECUTE THIS FILE ONE TIME ONLY """

import os
import numpy as np
import shutil
import random

classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "space"]

root_dir = "C:/code/gesture/asl_seg_rev/"
train_ratio = 0.60
test_ratio = 0.40

# create train and test folders
for classe in classes:
    os.makedirs(root_dir + "train/" + classe)
    os.makedirs(root_dir + "test/" + classe)

# split and copy images to train test folders
for classe in classes:
    src = root_dir + classe
    all_images = os.listdir(src)
    np.random.shuffle(all_images)
    train_image, test_image = all_images[:601], all_images[601:]
    for image in train_image:
        shutil.copy(src + "/" + image,
                    root_dir + "train/" + classe)
    for image in test_image:
        shutil.copy(src + "/" + image,
                    root_dir + "test/" + classe)


# remove old folders
for classe in classes:
    if os.path.exists(root_dir + classe):
        shutil.rmtree(root_dir + classe)
    else:
        print("folder does not exist")
