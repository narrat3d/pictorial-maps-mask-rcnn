'''
input: 
test images, i.e. pictorial maps with characters

output: 
detected bounding boxes and masks on the map

purpose: 
qualitative and quantitative evaluation of this CNN
'''
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from character_segmentation import config
from character_segmentation.inference import initialise_sessions, infer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask
from character_segmentation.config import mkdir_if_not_exists


def draw_bounding_box(image_draw, bounding_box, color):
    min_x, min_y, max_x, max_y = bounding_box
     
    for stroke_width in range(0, 3):
        image_draw.rectangle([min_x - stroke_width, min_y - stroke_width, 
                              max_x + stroke_width, max_y + stroke_width], outline=color)


def visualize_detections(image_pil, bounding_boxes, masks, scores, classes, color):
    image_draw = ImageDraw.Draw(image_pil)

    for bounding_box, image_mask_resized, score, class_ in zip(bounding_boxes, masks, scores, classes):
        if (score > config.INFERENCE_THRESHOLD and class_ == config.PERSON_CATEGORY_ID):     
            draw_bounding_box(image_draw, bounding_box, color)
            
            transparent_mask = Image.new('RGBA', image_pil.size)
            transparent_image = Image.new('RGBA', image_pil.size, (128, 0, 255, 100))
            transparent_mask.paste(transparent_image, image_mask_resized)
    
            image_pil.paste(transparent_mask, transparent_mask)
            # image_pil.paste(transparent_image, image_mask) does not work 


def load_coco_data():
    coco_data = json.load(open(config.COCO_GROUND_TRUTH_PATH))
    coco_images = coco_data["images"]
    image_ids = {}
    
    for coco_image in coco_images:
        image_ids[coco_image["file_name"]] = coco_image["id"]
        
    return image_ids


def calculate_coco_results(coco_results, image_id, masks, scores, classes):
    for mask, score, class_ in zip(masks, scores, classes):
        if (class_ != config.PERSON_CATEGORY_ID):
            continue
        
        # binary_mask = mask.point(lambda p: (p == 255 and 1) or 0)
        mask_np = np.asfortranarray(mask)
        
        mask_rle = COCOmask.encode(mask_np)
        mask_rle["counts"] = mask_rle["counts"].decode('utf-8')
        
        coco_results.append({
            "image_id": image_id, 
            "category_id": config.PERSON_CATEGORY_ID, 
            "segmentation": mask_rle, 
            "score": score,        
        })
        
    return coco_results


def print_coco_results(coco_results_path):
    coco_ground_truth = COCO(config.COCO_GROUND_TRUTH_PATH)
    coco_results = coco_ground_truth.loadRes(coco_results_path)
    
    coco_data = json.load(open(coco_results_path))
    image_ids = list(set(map(lambda a: a["image_id"], coco_data)))
    
    cocoEval = COCOeval(coco_ground_truth, coco_results, "segm")
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print(cocoEval.stats)


def evaluate(image_input_folder, inference_model_path, image_output_folder, coco_results_path):
    mkdir_if_not_exists(image_output_folder)
    
    image_names = os.listdir(image_input_folder)
    image_ids = load_coco_data()
    
    coco_results = []

    (image_session, image_tensor, image_placeholder,
      detection_session, detection_tensor, detection_placeholder) = initialise_sessions(inference_model_path)

    for image_name in image_names:
        print(image_name)
        image_file_path = os.path.join(image_input_folder, image_name)
           
        (detection_bounding_boxes, detection_masks, detection_scores, detection_classes, image) = \
            infer(image_session, image_tensor, image_placeholder, 
                  detection_session, detection_tensor, detection_placeholder, 
                  image_file_path) 
        
        image_id = image_ids[image_name]
        
        visualize_detections(image, detection_bounding_boxes, detection_masks, detection_scores, detection_classes, (0,255,255,0))
        calculate_coco_results(coco_results, image_id, detection_masks, detection_scores, detection_classes)
        
        image_name_without_ext = os.path.splitext(image_name)[0]
        image.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))
        
    json.dump(coco_results, open(coco_results_path, "w"))
   
    
if __name__ == '__main__':
    image_input_folder = os.path.join(config.TEST_DATA_PATH, "images")
    
    inference_model_name = "mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28"
    inference_model_path = config.get_original_inference_model_path(inference_model_name)
    
    image_output_folder = os.path.join(config.LOG_FOLDER, "results", inference_model_name)
    coco_results_path = os.path.join(config.LOG_FOLDER, "results", inference_model_name, "coco_results.json")
    
    evaluate(image_input_folder, inference_model_path, image_output_folder, coco_results_path)
    print_coco_results(coco_results_path)