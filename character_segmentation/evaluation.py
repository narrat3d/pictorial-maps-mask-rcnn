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
            
            # image_pil.paste(transparent_image, image_mask) does not work 
            transparent_mask = Image.new('RGBA', image_pil.size)
            transparent_image = Image.new('RGBA', image_pil.size, (128, 0, 255, 100))
            transparent_mask.paste(transparent_image, image_mask_resized)
    
            image_pil.paste(transparent_mask, transparent_mask)        


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
        
        mask_np = np.asfortranarray(mask)
        
        mask_rle = COCOmask.encode(mask_np)
        # bbox = COCOmask.toBbox(mask_rle)
        # area = COCOmask.area(mask_rle)
        
        mask_rle["counts"] = mask_rle["counts"].decode('utf-8')
        
        coco_results.append({
            "image_id": image_id, 
            "category_id": config.PERSON_CATEGORY_ID, 
            "segmentation": mask_rle, 
            "score": score,
            # "iscrowd": 0, 
            # "bbox": bbox.tolist(),
            # "area": area.item()    
        })
        
    return coco_results


def print_coco_results(coco_results_path, image_ids):
    coco_ground_truth = COCO(config.COCO_GROUND_TRUTH_PATH)
    coco_results = coco_ground_truth.loadRes(coco_results_path) 
    
    cocoEval = COCOeval(coco_ground_truth, coco_results, "segm")
    cocoEval.params.imgIds = image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    print(cocoEval.stats)
    
    
def filter_coco_results(coco_results_path, coco_filtered_results_path, category_ids_to_keep):
    coco_results = json.load(open(coco_results_path))
    
    filtered_coco_results = list(filter(lambda coco_result: coco_result["category_id"] in category_ids_to_keep and coco_result["score"] > 0.3,
                                 coco_results))

    json.dump(filtered_coco_results, open(coco_filtered_results_path, "w"))


def convert_image_file_names_to_ids(coco_results_path):
    coco_results = json.load(open(coco_results_path))

    image_ids = load_coco_data()
    
    for coco_result in coco_results:
        file_name = os.path.basename(coco_result["image_id"])
        coco_result["image_id"] = image_ids[file_name]
    
    json.dump(coco_results, open(coco_results_path, "w"))
    
    
def evaluate(image_input_folder, inference_model_path, image_output_folder, coco_results_path, allowed_image_ids):
    mkdir_if_not_exists(image_output_folder)
    
    image_names = os.listdir(image_input_folder)
    image_ids = load_coco_data()
    
    coco_results = []

    for image_name in image_names:
        image_id = image_ids.get(image_name)
        
        if (image_id is None or not image_id in allowed_image_ids):
            continue
        
        print(image_name)
        
        image_file_path = os.path.join(image_input_folder, image_name)
        image = Image.open(image_file_path)

        (image_session, image_tensor, image_placeholder,
            detection_session, detection_tensor, detection_placeholder) = initialise_sessions(inference_model_path, image)
    
        (detection_bounding_boxes, detection_masks, detection_scores, detection_classes, image) = \
            infer(image_session, image_tensor, image_placeholder, 
                  detection_session, detection_tensor, detection_placeholder, 
                  image_file_path) 
        
        visualize_detections(image, detection_bounding_boxes, detection_masks, detection_scores, detection_classes, (0,255,255,0))
        calculate_coco_results(coco_results, image_id, detection_masks, detection_scores, detection_classes)
        
        image_name_without_ext = os.path.splitext(image_name)[0]
        image.save(os.path.join(image_output_folder, image_name_without_ext + ".png"))
        
    json.dump(coco_results, open(coco_results_path, "w"))


def evaluate_original_model():
    # coco results json was created with train_and_eval.py with export_path specified
    coco_results_path = r"E:\CNN\logs\mask_rcnn\character_tensorflow\original_model\coco_results.json"
    coco_filtered_results_path = r"E:\CNN\logs\mask_rcnn\character_tensorflow\original_model\filtered_coco_results.json"
    filter_coco_results(coco_results_path, coco_filtered_results_path, [config.PERSON_CATEGORY_ID])
    convert_image_file_names_to_ids(coco_filtered_results_path)
    print_coco_results(coco_filtered_results_path, image_ids)


def evaluate_retrained_model():
    image_input_folder = os.path.join(config.TEST_DATA_PATH, "images")
    
    model_name = "1st_run_separated_stride8_0.25_0.5_1.0_2.0"
    step = 2304
    inference_model_path = config.get_inference_model_path(model_name, step)
    
    image_output_folder = os.path.join(config.LOG_FOLDER, model_name, "results-%s" % step)
    coco_results_path = os.path.join(config.LOG_FOLDER, model_name, "results-%s" % step, "coco_results.json")

    evaluate(image_input_folder, inference_model_path, image_output_folder, coco_results_path, image_ids)
    print_coco_results(coco_results_path, image_ids)
    
    
if __name__ == '__main__':
    image_ids = range(1, 53)

    evaluate_original_model()
    evaluate_retrained_model()