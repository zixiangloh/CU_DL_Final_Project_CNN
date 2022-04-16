import os
import re
import cv2
import numpy as np
import pickle as pkl

GEN_IMAGE = True

# Before we use this file to generate the biased car sheared image augmentation set, we will need to first
# download the original biased_cars dataset. This is because this code uses that dataset as a basis to
# generate a similar dataset with sheared images as discrete classes


def map_text_file_list_to_labels(txt_file_list, att_dict, debug=False):
    file_counter = 0
    phase_counter_dict = {'train': 0, 'val': 0, 'test': 0}
    unique_path_color = list()
    array_of_labels = list()
    array_of_image_paths = list()
    file_lists = list()
    with open(txt_file_list, 'r') as file:
        for line in file:
            file_counter += 1
            line = line.rstrip()
            file_lists.append(line)
            path_splits = line.split('/')
            path_splits_color = path_splits[1]
            unique_path_color.append(path_splits_color)
            path_splits_phase = path_splits[2]
            phase_counter_dict[path_splits_phase] = phase_counter_dict[path_splits_phase] + 1
            path_splits_image = path_splits[-1]
            if path_splits_image not in att_dict:
                raise Exception("Key {} not found!".format(path_splits_image))
            multi_label = att_dict[path_splits_image]
            array_of_labels.append(multi_label)
            array_of_image_paths.append(line)

    array_of_labels = np.array(array_of_labels)
    array_of_image_paths = np.array(array_of_image_paths)
    num_rows, num_cols = array_of_labels.shape
    unique_path_color = np.array(unique_path_color)

    print("\nNum files:", file_counter)
    print("Distribution of phases:", phase_counter_dict)
    print("Length unique path_splits_color:", np.unique(unique_path_color).shape)
    print("Label array shape:", array_of_labels.shape)
    for idx in range(num_cols):
        print("Unique elements in label array, col[{}]:".format(idx), np.unique(array_of_labels[:, idx]))

    # Use this for label to viewpoint/category correlation purposes
    if debug:
        pick_col = 1
        unique_color_label = np.unique(array_of_labels[:, pick_col])
        for color_label in unique_color_label:
            print("Col {:d} Label: {:d}".format(pick_col, color_label))
            unique_indices = np.where(array_of_labels[:, pick_col] == color_label)
            unique_path_color_slice = unique_path_color[unique_indices]
            unique_image_paths = array_of_image_paths[unique_indices]
            vals, counts = np.unique(unique_path_color_slice, return_counts=True)
            print("Values:", vals)
            print("Counts:", counts)
            print(unique_image_paths[0])
            print(unique_image_paths[1])
            # print("For color label {}:".format(color_label), unique_path_color_slice)
            # print("Image paths color label {}:".format(color_label), unique_image_paths)

    assert array_of_labels.shape[0] == file_counter, "Invalid dimension detected in labels array " \
                                                     "while running file {}".format(txt_file_list)
    assert len(file_lists) == file_counter, "Invalid dimension detected in file lists " \
                                            "while running file {}".format(txt_file_list)

    return file_lists, array_of_labels

# We know after some amount of checks that the labels are ordered in the following manner:
# [Scale, Rotation + Shear, Color, Model]
# We will now use this to our advantage to generate what we need for sheared car images.
# The target label is Rotation. To make things simple, we will use the existing labels to guide us.


# 5 Angles of shearing (Discretized)
def discrete_shear_image(input_image_path, output_file_name, cat=0):
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, dim = img.shape

    # Dictionary for labeling
    lookup_augmentation = {0: "Default",  # No augmentation
                           1: "Shear_x_0p25",  # X shear 0.25
                           2: "Shear_x_-0p25",  # X shear -0.25
                           3: "Shear_y_0p25",  # Y shear 0.25
                           4: "Shear_y_-0p25"}  # Y shear -0.25

    if cat == 0:
        m = np.float32([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    elif cat == 1:
        m = np.float32([[1, 0.25, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    elif cat == 2:
        m = np.float32([[1, -0.25, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
    elif cat == 3:
        m = np.float32([[1, 0, 0],
                        [0.25, 1, 0],
                        [0, 0, 1]])
    elif cat == 4:
        m = np.float32([[1, 0, 0],
                        [-0.25, 1, 0],
                        [0, 0, 1]])
    else:
        raise Exception("Invalid label detected: {:d}".format(cat))

    sheared_img = cv2.warpPerspective(img, m, (int(cols*1), int(rows*1)))  # Maintain the input shape
    sheared_img = cv2.cvtColor(sheared_img, cv2.COLOR_RGB2BGR)

    dir, _ = os.path.split(output_file_name)
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    cv2.imwrite(output_file_name, sheared_img)


# Generator function for mapping images to labels
def generate_shear_images(input_files, labels, txt_file):
    for file_idx, file in enumerate(input_files):
        # We are going to reprocess the file name
        out_file = file.split('/')
        in_file = os.path.join(*out_file[1:])
        in_file = os.path.join(biased_cars_root_path, in_file)
        out_file[0] = 'biased_cars_sheared'
        out_file[1] = out_file[1].replace('ROTATION', 'SHEAR')
        out_file[2] = out_file[2]  # Keep this. Should be either train, val or test
        out_file[3] = out_file[3]  # Keep this. Should be images.
        out_file[4] = out_file[4]  # Keep this. We will keep the file names the same.

        # Remember, we are going to use the label Rotation to guide us: [Scale, Rotation, Color, Model]
        target_label = labels[file_idx, 1]  # 1st position label
        discrete_shear_image(in_file, os.path.join(*out_file[1:]), cat=target_label)

        # We are going to write to a text file list
        if not os.path.isdir(biased_cars_sheared_dataset_lists_root_path):
            os.makedirs(biased_cars_sheared_dataset_lists_root_path, exist_ok=True)
        if file_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        out_txt_file = txt_file.replace('rotation', 'shear')
        with open(os.path.join(biased_cars_sheared_dataset_lists_root_path, out_txt_file), mode) as F:
            F.write('/'.join(out_file) + "\n")


this_path = os.path.dirname(os.path.abspath(__file__))

biased_cars_root_path = os.path.join(this_path, "..", "biased_cars")  # original data path

red_cars_path = os.path.join(biased_cars_root_path, "RED_CARS")  # Red cars path
red_cars_rotation_path_list = [os.path.join(biased_cars_root_path, "RED_CARS_9_SCALE_{:d}_ROTATION".format(i))
                               for i in range(5)]
black_cars_path = os.path.join(biased_cars_root_path, "BLACK_CARS")  # Black cars path
black_cars_rotation_path_list = [os.path.join(biased_cars_root_path, "BLACK_CARS_9_SCALE_{:d}_ROTATION".format(i))
                                 for i in range(5)]
green_cars_path = os.path.join(biased_cars_root_path, "GREEN_CARS")  # Green cars path
green_cars_rotation_path_list = [os.path.join(biased_cars_root_path, "GREEN_CARS_9_SCALE_{:d}_ROTATION".format(i))
                                 for i in range(5)]
blue_cars_path = os.path.join(biased_cars_root_path, "BLUE_CARS")  # Blue cars path
blue_cars_rotation_path_list = [os.path.join(biased_cars_root_path, "BLUE_CARS_9_SCALE_{:d}_ROTATION".format(i))
                                for i in range(5)]

# Stat and count number of files
print("RED_CARS train folder file count:", len(os.listdir(os.path.join(red_cars_path, "train", "images"))))
print("RED_CARS_Rotation train folder file count:", [len(os.listdir(os.path.join(i, "train", "images")))
                                                     for i in red_cars_rotation_path_list])
print("BLACK_CARS train folder file count:", len(os.listdir(os.path.join(black_cars_path, "train", "images"))))
print("BLACK_CARS_Rotation train folder file count:", [len(os.listdir(os.path.join(i, "train", "images")))
                                                       for i in black_cars_rotation_path_list])
print("GREEN_CARS train folder file count:", len(os.listdir(os.path.join(green_cars_path, "train", "images"))))
print("GREEN_CARS_Rotation train folder file count:", [len(os.listdir(os.path.join(i, "train", "images")))
                                                       for i in green_cars_rotation_path_list])
print("BLUE_CARS train folder file count:", len(os.listdir(os.path.join(blue_cars_path, "train", "images"))))
print("BLUE_CARS_Rotation train folder file count:", [len(os.listdir(os.path.join(i, "train", "images")))
                                                      for i in blue_cars_rotation_path_list])

source_att_file_path = os.path.join(biased_cars_root_path, "att_dict_simplified.p")  # original att dict
if not os.path.isfile(source_att_file_path):
    raise Exception("Att source file not found!!!")
else:
    with open(source_att_file_path, 'rb') as infile:
        source_att_dict = pkl.load(infile)
source_att_dict_keys = list(source_att_dict.keys())
print("Number of keys in source att_simplified_dict.p:", len(source_att_dict_keys))
source_att_dict_single_key = source_att_dict_keys[0]
print("Type of an element with key {} in source att_dict_simplified.p: {}".format(source_att_dict_single_key,
                                                                                  type(source_att_dict[
                                                                                           source_att_dict_single_key]))
      )
print("Key {}: {}".format(source_att_dict_single_key, source_att_dict[source_att_dict_single_key]))

# Check dataset_lists source files
biased_cars_dataset_lists_root_path = os.path.join(this_path, "..", "..", "dataset_lists", "biased_cars_lists")
biased_cars_sheared_dataset_lists_root_path = os.path.join(this_path, "..", "..", "dataset_lists",
                                                           "biased_cars_sheared_lists")

# Get dataset_lists text files
dataset_lists_txtfiles = [x for x in os.listdir(biased_cars_dataset_lists_root_path) if x.endswith('.txt')]

# Get training dataset
train_dataset_lists_txtfiles = [x for x in dataset_lists_txtfiles if x.startswith('train')]

# Get validation dataset
val_dataset_lists_txtfiles = [x for x in dataset_lists_txtfiles if x.startswith('val')]

# Get testing dataset
test_dataset_lists_txtfiles = [x for x in dataset_lists_txtfiles if x.startswith('test')]

for txt_file in train_dataset_lists_txtfiles:
    print(txt_file)
    input_files, labels = map_text_file_list_to_labels(os.path.join(biased_cars_dataset_lists_root_path, txt_file),
                                                       source_att_dict)
    if GEN_IMAGE:
        generate_shear_images(input_files, labels, txt_file)

for txt_file in val_dataset_lists_txtfiles:
    print(txt_file)
    input_files, labels = map_text_file_list_to_labels(os.path.join(biased_cars_dataset_lists_root_path, txt_file),
                                                       source_att_dict)
    if GEN_IMAGE:
        generate_shear_images(input_files, labels, txt_file)

for txt_file in test_dataset_lists_txtfiles:
    print(txt_file)
    input_files, labels = map_text_file_list_to_labels(os.path.join(biased_cars_dataset_lists_root_path, txt_file),
                                                       source_att_dict)
    if GEN_IMAGE:
        generate_shear_images(input_files, labels, txt_file)

