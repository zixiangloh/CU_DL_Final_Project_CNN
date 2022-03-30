import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import struct
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate


imagefile = 'train-images-idx3-ubyte'
labelfile = 'train-labels-idx1-ubyte'

with open(imagefile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}, NRows: {:d}, NCols: {:d}".format(magic, size, nrows, ncols))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

# plt.imshow(data[0, :, :], cmap=plt.cm.gray_r)
# plt.show()

with open(labelfile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}".format(magic, size))
    data_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_labels = data_labels.reshape((size, ))

print(data_labels[0])

view_point_classes = list(range(0, 10))
view_point_angles = list(range(0, 360, 36))

source_path = os.path.join("..", "mnist_rotation", "self_generated")
folder_lists = [f for f in os.listdir(source_path) if not f.startswith('.')]

print(folder_lists)


def save_img(filename, array):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for x in range(28):
        for y in range(28):
            pix[x, y] = int(array[y][x])
    img.save(filename)


# This code generates the images into folders and organizes them into the same format as the original paper.
# We can use this to easily generate a dataset that can be used to immediately run the demo code for increasing data
# diversity, but instead of using MNIST handwriting, we use MNIST Clothing.
# You need to first download the mnist_rotation source data set from the author. Because the code uses that folder as
# a reference.
# The file name for each png file is named {category}_{azimuth}_{index}.png. The category is one of 10 classes in the
# MNIST clothing dataset. The Azimuth is a rotation angle discretized between 0 to 360. To make it difficult for the
# Neural Network, we can also try randomly assigning an integer of rotation between each discretized class.
for f in folder_lists:
    print(f)
    for i in range(10):
        match_string = r'^{:d}_*'.format(i)
        file_lists = [file for file in os.listdir(os.path.join(source_path, f)) if re.match(match_string, file)]
        m = re.match(r'(\d+)_(\d+)_*', file_lists[0])
        category, azimuth_cat = int(m.group(1)), int(m.group(2))
        azimuth = view_point_angles[azimuth_cat]
        print("Category: {:d}, Azimuth_cat: {:d}, Azimuth: {:d}".format(category, azimuth_cat,
                                                                        azimuth))
        curr_path = os.path.join('.', "self_generated", f)
        if not os.path.isdir(curr_path):
            os.makedirs(curr_path, exist_ok=True)
        # We subset out all data with labels
        row_idx = np.where(data_labels == i)[0]
        # Get images where matched
        for j in row_idx:
            original_image = data[j, :, :]
            output_image = rotate(original_image, angle=azimuth)
            name = '{:d}_{:d}_{:d}.png'.format(category, azimuth_cat, j)
            outname = os.path.join(curr_path, name)
            save_img(outname, output_image)
