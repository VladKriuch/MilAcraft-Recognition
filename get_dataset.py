# Imports
import kaggle
import os

from local_utils import parse_xml, convert_to_yolov5, move_files_to_folder

import yaml

# Paths
ROOT_DIR = "data/"
IMAGE_DIR = ROOT_DIR + "JPEGImages/"
HORIZONTAL_BB = ROOT_DIR + "Annotations/Horizontal Bounding Boxes/"
ORIENTED_BB = ROOT_DIR + "Annotations/Oriented Bounding Boxes/"

TRAIN_SET_TXT = ROOT_DIR + "ImageSets/Main/train.txt"
TEST_SET_TXT = ROOT_DIR + "ImageSets/Main/test.txt"


def get_train_set_ids():
    with open(TRAIN_SET_TXT, 'r') as file:
        ids = file.read().splitlines()
    return ids


def get_test_set_ids():
    with open(TEST_SET_TXT, 'r') as file:
        ids = file.read().splitlines()
    return ids


if __name__ == "__main__":
    os.environ['KAGGLE_USERNAME'] = "vladkriuch"
    os.environ['KAGGLE_KEY'] = "19e06b35085d6f6a2e6be9797df30403"

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('khlaifiabilel/military-aircraft-recognition-dataset',
                                      path='data',
                                      unzip=True)
    print("Kaggle upload finished")

    train_set_ids = get_train_set_ids()
    test_set_ids = get_test_set_ids()
    print(f'Size of train set is: {len(train_set_ids)}')
    print(f'Size of test set is: {len(test_set_ids)}')
    print(f'Total images: {len(train_set_ids) + len(test_set_ids)}')

    # Create folders
    os.mkdir("data for yolo")
    os.mkdir("data for yolo/labels")
    os.mkdir("data for yolo/images")
    os.mkdir("data for yolo/labels/train")
    os.mkdir("data for yolo/labels/test")
    os.mkdir("data for yolo/images/train")
    os.mkdir("data for yolo/images/test")

    # Convert classes to classes_map for YOLO model
    classes = ['A19', 'A1', 'A20', 'A16', 'A5', 'A13', 'A15', 'A3', 'A17', 'A11',
               'A14', 'A8', 'A2', 'A10', 'A9', 'A4', 'A18', 'A7', 'A12', 'A6']

    classes_map = {item: index for index, item in enumerate(classes)}

    # Convert dataset to yolo format
    for obj_id in train_set_ids:
        info_dict = parse_xml(HORIZONTAL_BB + str(obj_id) + ".xml")
        convert_to_yolov5(info_dict, str(obj_id), "data for yolo/labels/train/", classes_map, IMAGE_DIR)

    print("Train set converted")

    for obj_id in test_set_ids:
        info_dict = parse_xml(HORIZONTAL_BB + str(obj_id) + ".xml")
        convert_to_yolov5(info_dict, str(obj_id), "data for yolo/labels/test/", classes_map, IMAGE_DIR)

    print("Test set converted")
    train_img_files = [IMAGE_DIR + str(obj_id) + ".jpg" for obj_id in train_set_ids]
    test_img_files = [IMAGE_DIR + str(obj_id) + ".jpg" for obj_id in test_set_ids]

    # Move the splits into their folders
    move_files_to_folder(train_img_files, 'data for yolo/images/train')
    move_files_to_folder(test_img_files, 'data for yolo/images/test')

    class_id_to_name_mapping = dict(zip(classes_map.values(), classes_map.keys()))

    # Make .yaml file for yolo
    d = {
        'path': '../data for yolo/',
        'train': 'images/train',
        'val': 'images/test',
        'nc': 20,
        'names': class_id_to_name_mapping
    }

    with open('dataset.yml', 'w') as yaml_file:
        yaml.dump(d, yaml_file, default_flow_style=False)