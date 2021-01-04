import tensorflow as tf
import json
import os
from PIL import Image
import numpy as np
from object_detection.utils import dataset_util
import io
from character_segmentation import config


IMAGE_SOURCE_FOLDER_NAME = "images"
MASK_SOURCE_FOLDER_NAME = "masks"
KEYPOINT_SOURCE_FOLDER_NAME = "keypoints"


#source: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]) + 1, np.min(a[1]), np.max(a[1]) + 1
    return bbox


def create_tf_example(image_source_folder, image_file_name, mask_file_path, keypoints_file_path, cropped_regions_data_folder):    
    image_file_path = os.path.join(image_source_folder, image_file_name)
    encoded_filename = image_file_path.encode()
    
    with tf.gfile.GFile(image_file_path, 'rb') as gfile:
        encoded_image_data = gfile.read()
    
    image = Image.open(io.BytesIO(encoded_image_data))
    # image.save(r"E:\CNN\logs\mask_rcnn\character_tensorflow\1st_run_separated_stride8_0.25_0.5_1.0_2.0_3epochs\train_results\record_image.png")
        
        
    with open(keypoints_file_path) as keypoints_file:
        keypoints_array = json.load(keypoints_file)
        number_of_characters = len(keypoints_array)
    
    mask = Image.open(mask_file_path)
    
    mask_array = np.asarray(mask)
    # extract values of the the first band (character count should start with 1) 
    mask_array = mask_array[:, :, 0]
    width, height = mask.size
    
    image_format = b"jpeg" # b'jpeg' or b'png'

    class_name = config.PERSON_CATEGORY_NAME.encode('utf-8')
    class_id = config.PERSON_CATEGORY_ID

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    encoded_mask_png_list = []
    
    for i in range(number_of_characters):
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/instance_segmentation.md
        is_character = mask_array == i
        is_background = mask_array != i
        
        if (np.count_nonzero(is_character) == 0):
            continue
        
        mask_array_instance = np.copy(mask_array)
        
        mask_array_instance[is_background] = 0
        mask_array_instance[is_character] = 1
        
        (ymin, ymax, xmin, xmax) = bbox(mask_array_instance)
        
        # source: https://github.com/priya-dwivedi/Deep-Learning/blob/master/Custom_Mask_RCNN/create_pet_tf_record.py
        img = Image.fromarray(mask_array_instance)
        output = io.BytesIO()
        img.save(output, format='PNG')
        img_bytes = output.getvalue()
        encoded_mask_png_list.append(img_bytes)
        
        # region_filename = file_name[:-4] + "_%s_mask" % i + file_name[-4:]
        # img.save(region_filename, format='PNG')

        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        
        classes_text.append(class_name)
        classes.append(class_id)
        
        cropped_region = image.crop((xmin, ymin, xmax, ymax))
        region_filename = image_file_name[:-4] + "_%s" % i + image_file_name[-4:]
        
        cropped_region_file_path = os.path.join(cropped_regions_data_folder, region_filename)
        cropped_region.save(cropped_region_file_path)
        

    if (len(classes) == None):
        return (None, 0)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_filename),
        'image/source_id': dataset_util.bytes_feature(encoded_filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/mask': dataset_util.bytes_list_feature(encoded_mask_png_list)
    }))
    return (tf_example, len(classes))


def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)


def main(_):
    modes = {}
    
    for dataset_name in config.RECORD_NAMES:
        modes[dataset_name] = {
            "source_folder": os.path.join(config.SOURCE_DATA_FOLDER, dataset_name),
            "record_name": "%s.record" % dataset_name
        }
        
    for mode_name, mode in modes.items():        
        record_path = os.path.join(config.DATA_FOLDER, mode["record_name"])
        mode["writer"] = tf.python_io.TFRecordWriter(record_path)
        
        mode["number_of_boxes"] = 0
        mode["number_of_examples"] = 0
        
        cropped_regions_data_folder = os.path.join(config.CROPPED_REGIONS_FOLDER, mode_name)
        mkdir_if_not_exists(cropped_regions_data_folder)
        
        image_source_folder = os.path.join(mode["source_folder"], IMAGE_SOURCE_FOLDER_NAME)
        
        for image_name in os.listdir(image_source_folder):            
            mask_file_path = os.path.join(mode["source_folder"], MASK_SOURCE_FOLDER_NAME, image_name.replace(".jpg", ".png"))
            keypoints_file_path = os.path.join(mode["source_folder"], KEYPOINT_SOURCE_FOLDER_NAME, image_name.replace(".jpg", ".json"))
            
            (tf_example, number_of_boxes) = create_tf_example(image_source_folder, image_name, 
                                                              mask_file_path, keypoints_file_path, cropped_regions_data_folder)
            
            if (tf_example == None):
                continue
            
            mode["writer"].write(tf_example.SerializeToString())
            mode["number_of_boxes"] += number_of_boxes
            mode["number_of_examples"] += 1

        mode["writer"].close()


    for mode_name, mode in modes.items():
        print(mode_name)
        print("number_of_examples:", mode["number_of_examples"])
        print("number_of_boxes:", mode["number_of_boxes"])


if __name__ == '__main__':
    tf.app.run()