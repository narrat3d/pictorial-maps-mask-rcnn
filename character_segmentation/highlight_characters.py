'''
input: 
test images, i.e. pictorial maps with characters

output: 
detected bounding boxes and masks on the map

purpose: 
qualitative evaluation of this CNN
'''
import os
from PIL import Image, ImageDraw
from character_segmentation import config
from character_segmentation.inference import initialise_sessions, infer

image_input_folder = r"E:\CNN\masks\data\separated_input"
inference_model_path = config.get_inference_model_path("1st_run_separated_stride8_0.25_0.5_1.0_2.0", 1000)
image_output_folder = r"E:\CNN\masks\data\separated_output"


def draw_bounding_box(image_draw, bounding_box, color):
    min_x, min_y, max_x, max_y = bounding_box
     
    for stroke_width in range(0, 3):
        image_draw.rectangle([min_x - stroke_width, min_y - stroke_width, 
                              max_x + stroke_width, max_y + stroke_width], outline=color)


def get_detection_mask_images(image_size, masks):
    detection_mask_images = []
    
    for mask in masks:
        image_mask = Image.fromarray(mask * 255, "L")
        image_mask_resized = image_mask.resize(image_size)
        
        detection_mask_images.append(image_mask_resized)
        
    return detection_mask_images


def visualize_detections(image_pil, bounding_boxes, masks, color):
    detection_mask_images = get_detection_mask_images(image_pil.size, masks)
    image_draw = ImageDraw.Draw(image_pil)

    for bounding_box, image_mask_resized in zip(bounding_boxes, detection_mask_images):
        draw_bounding_box(image_draw, bounding_box, color)
        
        transparent_mask = Image.new('RGBA', image_pil.size)
        transparent_image = Image.new('RGBA', image_pil.size, (128, 0, 255, 100))
        transparent_mask.paste(transparent_image, image_mask_resized)

        image_pil.paste(transparent_mask, transparent_mask)
        # image_pil.paste(transparent_image, image_mask) does not work 


def highlight_characters_on_maps(image_input_folder, inference_model_path, image_output_folder):
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
    
        visualize_detections(image, detection_bounding_boxes, detection_masks, (0,255,255,0))
        
        image_name_without_ext = os.path.splitext(image_name)[0]
        image.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))  
   
    
if __name__ == '__main__':
    highlight_characters_on_maps(image_input_folder, inference_model_path, image_output_folder)