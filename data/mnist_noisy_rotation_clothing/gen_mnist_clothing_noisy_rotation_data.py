import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import math
import struct
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage import rotate
import random


# Before running this code, you need to have the MNIST rotation data checked out from the original authors.
# This code uses the data folder generated from the MNIST rotation data set as a guide to generate a similar rotated
# data set using MNIST fashion (or MNIST clothing)

imagefile = os.path.join('..', 'mnist_clothing', 'train-images-idx3-ubyte')
labelfile = os.path.join('..', 'mnist_clothing', 'train-labels-idx1-ubyte')
imagefile_test = os.path.join('..', 'mnist_clothing', 't10k-images-idx3-ubyte')
labelfile_test = os.path.join('..', 'mnist_clothing', 't10k-labels-idx1-ubyte')

with open(imagefile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}, NRows: {:d}, NCols: {:d}".format(magic, size, nrows, ncols))
    data_train = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_train = data_train.reshape((size, nrows, ncols))

# plt.imshow(data[0, :, :], cmap=plt.cm.gray_r)
# plt.show()

with open(labelfile, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}".format(magic, size))
    data_train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_train_labels = data_train_labels.reshape((size, ))

print(data_train_labels[0])

with open(imagefile_test, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}, NRows: {:d}, NCols: {:d}".format(magic, size, nrows, ncols))
    data_test = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_test = data_test.reshape((size, nrows, ncols))

with open(labelfile_test, 'rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    print("Magic: {:d}, Size: {:d}".format(magic, size))
    data_test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data_test_labels = data_test_labels.reshape((size, ))

print(data_test_labels[0])

num_rotation_classes = 10
rotation_angle_discrete = 360 // num_rotation_classes

view_point_classes = list(range(0, num_rotation_classes))
view_point_angles = list(range(0, 360, rotation_angle_discrete))

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

noise_percent = 0.5  # This is the noise margin percentage.

for f in folder_lists:
    print(f)
    for i in range(num_rotation_classes):  # 10 classes
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
        # Where do we source the data from (test or train)
        if f == 'unseen_images':
            data_source = data_test
            data_label_source = data_test_labels
        else:
            data_source = data_train
            data_label_source = data_train_labels
        # We subset out all data with labels
        row_idx = np.where(data_label_source == i)[0]
        # Get images where matched
        for j in row_idx:
            original_image = data_source[j, :, :]
            # Each image is in discrete 36 degree angles + some noise
            # [0, 36, 72, 108, 144, 180, 216, 252, 288, 324]
            # Noise Margin = 36 * Percent Noise
            lower_noise_margin = math.floor(azimuth - noise_percent/2 * rotation_angle_discrete)
            upper_noise_margin = math.floor(azimuth + noise_percent/2 * rotation_angle_discrete)
            azimuth_with_noise = random.randint(lower_noise_margin, upper_noise_margin)
            output_image = rotate(original_image, angle=azimuth_with_noise)
            name = '{:d}_{:d}_{:d}.png'.format(category, azimuth_cat, j)
            outname = os.path.join(curr_path, name)
            save_img(outname, output_image)

# Read from source texts and see what's going on:
source_text_path = os.path.join('..', '..', 'dataset_lists', 'mnist_rotation_lists')
source_text_lists = [f for f in os.listdir(source_text_path) if f.endswith('.txt')]

print(source_text_lists)

dest_path = os.path.join('..', '..', 'dataset_lists', 'mnist_noisyrotclothing_lists')
if not os.path.isdir(dest_path):
    os.makedirs(dest_path, exist_ok=True)

for file in source_text_lists:
    print(file)
    # We deal with files with 'tests' since they are straightforward.
    # The text files are all identical
    if file.startswith('test'):
        with open(os.path.join(source_text_path, file), 'r') as f:
            lines_list = f.readlines()
            print("Num lines:", len(lines_list))
            unique_dirname = set()
            unique_png_filename = set()
            for line in lines_list:
                _, _, dirname, png_filename = line.split("/")
                unique_dirname.add(dirname)
                unique_png_filename.add(png_filename)
            print(unique_dirname)
            print(len(unique_png_filename))
            file_out = file.replace('rotation', 'noisyrotclothing')
        with open(os.path.join(dest_path, file_out), 'w') as f_out:
            # Get all the files from unseen images
            outgoing_files = os.listdir(os.path.join('.', "self_generated", 'unseen_images'))
            for outgoing_file in outgoing_files:
                f_out.write('mnist_noisy_rotation_clothing/self_generated/unseen_images/' + outgoing_file + '\n')

    if file.startswith('train'):
        # We open the training file
        with open(os.path.join(source_text_path, file), 'r') as f:
            lines_list = f.readlines()
            # print("Num lines:", len(lines_list))
            unique_png_filename_train = dict()
            for line in lines_list:
                _, _, dirname, png_filename = line.split("/")
                if dirname not in unique_png_filename_train:
                    unique_png_filename_train[dirname] = [png_filename]
                else:
                    unique_png_filename_train[dirname].append(png_filename)
        # We open the validation file
        val_file = file.replace('train', 'val')
        with open(os.path.join(source_text_path, val_file), 'r') as f:
            lines_list = f.readlines()
            # print("Num lines:", len(lines_list))
            unique_png_filename_val = dict()
            for line in lines_list:
                _, _, dirname, png_filename = line.split("/")
                if dirname not in unique_png_filename_val:
                    unique_png_filename_val[dirname] = [png_filename]
                else:
                    unique_png_filename_val[dirname].append(png_filename)
        # Split by number
        master_train_files = list()
        master_val_files = list()
        for key in unique_png_filename_train:
            train_num = len(unique_png_filename_train[key])
            val_num = len(unique_png_filename_val[key])
            this_dir = os.path.join('.', "self_generated", key)
            file_list = os.listdir(this_dir)
            random.shuffle(file_list)
            train_files = file_list[:train_num]
            val_files = file_list[train_num:train_num+val_num]
            assert len(train_files) == train_num, "Training file count mismatch!"
            assert len(val_files) == val_num, "Validation file count mismatch!"
            train_files = ['mnist_noisy_rotation_clothing/self_generated/' + key + '/' + x for x in train_files]
            val_files = ['mnist_noisy_rotation_clothing/self_generated/' + key + '/' + x for x in val_files]
            master_train_files.extend(train_files)
            master_val_files.extend(val_files)

        file_out = file.replace('rotation', 'noisyrotclothing')
        with open(os.path.join(dest_path, file_out), 'w') as f_out:
            for outgoing_file in master_train_files:
                f_out.write(outgoing_file + '\n')

        val_file_out = val_file.replace('rotation', 'noisyrotclothing')
        with open(os.path.join(dest_path, val_file_out), 'w') as f_out:
            for outgoing_file in master_val_files:
                f_out.write(outgoing_file + '\n')
