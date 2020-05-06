'''
input: 
training or validation data containing maps with characters

output: 
detected characters on the map with the complete background or 
the background within the detection mask

purpose: 
the output could be used as training data for another CNN
which detects body parts and/or joints
'''
import os
import json
import numpy as np
from character_segmentation import config
from PIL import Image
from character_segmentation.inference import initialise_sessions, infer


input_folder = r"E:\CNN\masks\data\character_maps\separated"
inference_model_path = config.get_inference_model_path("1st_run_separated_stride8_0.25_0.5_1.0_2.0", 1000)
output_folder = r"E:\CNN\masks\data\separated_output"


instance_counter = 0


def shift_keypoints(keypoints, point):
    keypoints_shifted = {}
    
    for key, coords in keypoints.items():
        keypoints_shifted[key] = [coords[0] + point[0], coords[1] + point[1]]
        
    return keypoints_shifted


def filter_keypoints(keypoints, width, height):
    for key, coords in zip(list(keypoints.keys()), list(keypoints.values())):
        if (coords[0] < 0 or coords[0] > width or coords[1] < 0 or coords[1] > height):
            del keypoints[key]


def get_body_part_instances(mask_file_path, keypoint_file_path):
    body_part_instances = []
    bboxes = []
    instance_mask_images = []
    
    mask_map_image = Image.open(mask_file_path)
    instances_image = mask_map_image.getchannel(0)
    body_parts_image = mask_map_image.getchannel(1)
    
    keypoints = json.load(open(keypoint_file_path))
    number_of_characters = len(keypoints)
    
    for i in range(1, number_of_characters + 1):
        instance_mask_image = Image.eval(instances_image, lambda x: ((x == i) and 255) or 0) 

        instance_body_parts_image = Image.new("L", body_parts_image.size, 255)
        instance_body_parts_image.paste(body_parts_image, instance_mask_image)
        
        body_part_instances.append(instance_body_parts_image)
        bboxes.append(instance_mask_image.getbbox())
        instance_mask_images.append(np.array(instance_mask_image, np.float) / 255.0)
        
    return (body_part_instances, keypoints, bboxes, instance_mask_images)


def match_instances(image, gt_masks, keypoints, detected_mask_images, detected_bboxes,
                    instance_image_output_folder, instance_image_masked_output_folder,
                    instance_mask_output_folder, instance_keypoints_output_folder):
    global instance_counter
    
    for detected_mask_image, bbox in zip(detected_mask_images, detected_bboxes):
        max_overlap = 0
        max_overlap_index = None
        
        for index, gt_mask in enumerate(gt_masks):
            instance_mask_image = Image.eval(gt_mask, lambda x: ((x != 255) and 255) or 0) 
            
            mask_union = Image.new("L", detected_mask_image.size)
            mask_union.paste(instance_mask_image, detected_mask_image)
            
            overlap = sum(list(mask_union.getdata()))
            
            if (overlap > max_overlap):
                max_overlap = overlap
                max_overlap_index = index
        
        if (max_overlap_index is None):
            continue
        
        matched_gt_mask = gt_masks[max_overlap_index]
        matched_keypoints = keypoints[max_overlap_index]
        
        image_crop = image.crop(bbox)
        detected_mask_crop = detected_mask_image.crop(bbox)
        
        image_crop_masked = Image.new("RGB", image_crop.size, (255, 255, 255))
        image_crop_masked.paste(image_crop, detected_mask_crop)        
        
        gt_mask_crop = matched_gt_mask.crop(bbox)
        
        shifted_keypoints = shift_keypoints(matched_keypoints, (-bbox[0], -bbox[1]))
        filter_keypoints(shifted_keypoints, image_crop.width, image_crop.height)
        
        instance_file_name = "%s.jpg" % instance_counter
        image_crop.save(os.path.join(instance_image_output_folder, instance_file_name))
        image_crop_masked.save(os.path.join(instance_image_masked_output_folder, instance_file_name))
        gt_mask_crop.save(os.path.join(instance_mask_output_folder, instance_file_name))
        json.dump(shifted_keypoints, open(os.path.join(instance_keypoints_output_folder, "%s.json" % instance_counter), "w"))
        
        instance_counter += 1


def extract_characters_from_maps(input_folder, inference_model_path, output_folder):
    image_input_folder = os.path.join(input_folder, "images")
    mask_input_folder = os.path.join(input_folder, "masks")
    keypoint_input_folder = os.path.join(input_folder, "keypoints")
    
    instance_image_masked_output_folder = os.path.join(output_folder, "images_with_some_background")
    instance_image_output_folder = os.path.join(output_folder, "images_with_background")
    instance_mask_output_folder = os.path.join(output_folder, "masks")
    instance_keypoints_output_folder = os.path.join(output_folder, "keypoints")
    
    list(map(config.mkdir_if_not_exists, [output_folder, instance_image_masked_output_folder, instance_image_output_folder, 
                                   instance_mask_output_folder, instance_keypoints_output_folder]))
    
    image_names = os.listdir(image_input_folder)

    (image_session, image_tensor, image_placeholder,
      detection_session, detection_tensor, detection_placeholder) = initialise_sessions(inference_model_path)

    for image_name in image_names:
        print(image_name)
        image_file_path = os.path.join(image_input_folder, image_name)
           
        (detection_bounding_boxes, detection_masks, image) = \
            infer(image_session, image_tensor, image_placeholder, 
                  detection_session, detection_tensor, detection_placeholder, 
                  image_file_path) 
    
        image_name_without_ext = os.path.splitext(image_name)[0]
        mask_file_path = os.path.join(mask_input_folder, image_name_without_ext + ".png")
        keypoint_file_path = os.path.join(keypoint_input_folder, image_name_without_ext + ".json")
        
        (body_part_instances, keypoint_instances, _, _) = get_body_part_instances(mask_file_path, keypoint_file_path)        
        match_instances(image, body_part_instances, keypoint_instances, detection_masks, detection_bounding_boxes,
                        instance_image_output_folder, instance_image_masked_output_folder, instance_mask_output_folder, 
                        instance_keypoints_output_folder)
        

if __name__ == '__main__':
    extract_characters_from_maps(input_folder, inference_model_path, output_folder)