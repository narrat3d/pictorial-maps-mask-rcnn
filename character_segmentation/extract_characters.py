'''
input: 
test images, i.e. pictorial maps with characters

output:
folder named after the test image
extracted_map.png - map without characters
config.json - metadata file containing the size of the map and the number of detected characters

folders with extracted characters (char<number>)
image.png - extracted character, padded to square size
padding.json - how much the image was relatively padded in x- and y-direction
topLeftCorner.json - where the character was absolutely positioned on the map (x - left, y - top)

purpose:
passing the characters to another CNN extracting body parts and joints
afterwards characters could be animated on a website 
'''
from character_segmentation import config
from character_segmentation.inference import initialise_sessions, infer
import os
from PIL import Image
import shutil
import json


image_input_folder = r"E:\CNN\masks\data\separated_input"
output_folder = r"E:\CNN\masks\data\separated_output"
inference_model_path = config.get_inference_model_path("1st_run_separated_stride8_0.25_0.5_1.0_2.0", 1000)


def pad_image(image):
    if (image.width > image.height):
        length = image.width
        x_offset = 0
        y_offset = round((length - image.height) / 2)
    else :
        length = image.height
        x_offset = round((length - image.width) / 2)
        y_offset = 0
        
    padded_image = Image.new("RGB", (length, length), (255, 255, 255))
    padded_image.paste(image, (x_offset, y_offset))
    
    return padded_image, [x_offset, y_offset]
    

def crop_and_mask(image, bbox, mask, alpha=False):
    image_crop = image.crop(bbox)
    mask_crop = mask.crop(bbox)
    
    if (alpha):
        image_crop_masked = Image.new("RGBA", image_crop.size)
    else :
        image_crop_masked = Image.new("RGB", image_crop.size, (255, 255, 255))
        
    image_crop_masked.paste(image_crop, mask_crop)
    
    return image_crop_masked


def extract_characters_from_map(image, bounding_boxes, masks):
    top_left_corners = []
    paddings = []
    detected_characters = []
    
    image_with_cropped_areas = image.copy()
    white_area = Image.new("RGB", image.size, (255, 255, 255))

    for bounding_box, mask in zip(bounding_boxes, masks):        
        image_crop_masked = crop_and_mask(image, bounding_box, mask)
        image_crop_masked_padded, padding = pad_image(image_crop_masked)
        
        detected_characters.append(image_crop_masked_padded)
        top_left_corners.append([bounding_box[0], bounding_box[1]])
        paddings.append(padding)
        
        image_with_cropped_areas.paste(white_area, mask)

    return detected_characters, top_left_corners, paddings, image_with_cropped_areas


if __name__ == '__main__':
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
            
        character_images, top_left_corners, paddings, image_with_cropped_areas = \
            extract_characters_from_map(image, detection_bounding_boxes, detection_masks)
        
        output_map_folder = os.path.join(output_folder, os.path.splitext(image_name)[0])
        
        if (os.path.exists(output_map_folder)):
            shutil.rmtree(output_map_folder)
        
        os.mkdir(output_map_folder)
        image_with_cropped_areas.save(os.path.join(output_map_folder, "extracted_map.jpg"))
        
        counter = 0
        
        for character_image, top_left_corner, padding in zip(character_images, top_left_corners, paddings):
            character_output_folder = os.path.join(output_map_folder, "char%s" % counter)
            os.mkdir(character_output_folder)
            
            character_image.save(os.path.join(character_output_folder, "image.jpg"))
            
            top_left_corner_json_path = os.path.join(character_output_folder, "topLeftCorner.json")
            json.dump(top_left_corner, open(top_left_corner_json_path, "w"))
            
            padding_json_path = os.path.join(character_output_folder, "padding.json")
            json.dump(padding, open(padding_json_path, "w"))
            
            counter += 1
            
        map_config = {
            "numberOfCharacters": counter,
            "width": image.width,
            "height": image.height
        }
        map_config_file_path = os.path.join(output_map_folder, "config.json")
        json.dump(map_config, open(map_config_file_path, "w"))