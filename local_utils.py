import xml.etree.ElementTree as ET

import os
import shutil

import cv2
# Reading xml files
# XML reader
def parse_xml(xml_file_path: str):
    # Parses xml file, returns dict
    tree = ET.parse(xml_file_path)

    filename = tree.find('filename').text
    img_size = {
        'width': int(tree.find('size').find('width').text),
        'height': int(tree.find('size').find('height').text),
        'depth': int(tree.find('size').find('depth').text)
    }
    segmented = int(tree.find('segmented').text)
    bboxes = []
    for obj in tree.findall('object'):
        bndbox = obj.find('bndbox')
        bboxes.append({
            'name': obj.find('name').text,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        })
    database = tree.find('source').find('database').text

    return {
        'filename': filename,
        'img_size': img_size,
        'segmented': segmented,
        'bboxes': bboxes,
        'database': database
    }


def get_xml_file_path(filename, HORIZONTAL_BB_PATH, ORIENTED_BB_PATH, option='HORIZONTAL'):
    if option == 'HORIZONTAL':
        return HORIZONTAL_BB_PATH + filename
    elif option == 'ORIENTED':
        return ORIENTED_BB_PATH + filename

# Helping functions
# Displaying images

def get_image(image_path: str):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def get_image_shape(image_path: str):
    img = cv2.imread(image_path)
    return img.shape


# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict, obj_id, save_folder, classes_map, IMAGE_DIR):
    print_buffer = []

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = classes_map[b["name"]]
        except KeyError:
            print("Invalid Class. Must be one from ", classes_map.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])

        # Normalise the co-ordinates by the dimensions of the image
        image_h, image_w, image_c = get_image_shape(IMAGE_DIR + obj_id + ".jpg")
        b_center_x /= image_w
        b_center_y /= image_h
        b_width    /= image_w
        b_height   /= image_h

        #Write the bbox details to the file
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save
    save_file_name = os.path.join(save_folder, obj_id + ".txt")

    # Save the annotation to disk
    # print("\n".join(print_buffer), file= open(save_file_name, "w"))
    with open(save_file_name, 'w') as fl:
      fl.write("\n".join(print_buffer))


# Copy images to YOLO
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False


# Visualization
def get_bounding_boxes(obj_id: str):
    fpath = get_xml_file_path(str(obj_id) + ".xml")
    xml_parsed = parse_xml(fpath)
    bounding_boxes = xml_parsed["bboxes"]

    return bounding_boxes


def draw_boxes(image, bounding_boxes: list):
    for bbox in bounding_boxes:
        image = cv2.rectangle(image, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (255,0,0), 2)
        cv2.putText(image, bbox['name'], (bbox['xmax'], bbox['ymax']+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,20,20), 3)

    return image