import tensorflow as tf
import numpy as np
import os
from character_segmentation import config
from object_detection import export_inference_graph

parameter_prefix = "DetectionMasks_Precision/"

coco_parameters = [
    "mAP",
    "mAP@.50IOU",
    "mAP@.75IOU",
    "mAP (small)",
    "mAP (medium)",
    "mAP (large)"
]
mAP_50_index = coco_parameters.index("mAP@.50IOU")

aggregated_results = {
}

coco_parameters = list(map(lambda p: parameter_prefix + p, coco_parameters))

def get_highest_coco_values(events_file_path, highest_coco_values, model_ckpt):
    current_values = None
    current_step = -1
    
    for e in tf.train.summary_iterator(events_file_path):
        if (e.step != current_step):
            current_values = [None] * 6
            current_step = e.step
        
        for v in e.summary.value:
            try :
                index = coco_parameters.index(v.tag)
            
                if (index == mAP_50_index):
                    if v.simple_value > highest_coco_values[mAP_50_index]:
                        highest_coco_values = current_values
                        model_ckpt = current_step
                    else :
                        break
                
                current_values[index] = v.simple_value
                
            except ValueError:
                pass
    
    return highest_coco_values, model_ckpt


def print_event_values(events_file_path):
    for e in tf.train.summary_iterator(events_file_path):
        print (e.step)
        
        for v in e.summary.value:
            print (v.tag, v.simple_value)


def main():
    folder_names = os.listdir(config.LOG_FOLDER)
    convert_model = False
    
    for folder_name in folder_names:
        eval_folder_path = os.path.join(config.LOG_FOLDER, folder_name, "eval")
        
        highest_coco_values = [0] * len(coco_parameters)
        model_ckpt = 0
        
        if (os.path.exists(eval_folder_path)):
            eval_file_names = os.listdir(eval_folder_path)
                        
            for eval_file_name in eval_file_names:
                if (eval_file_name.startswith("events")):
                    eval_file_path = os.path.join(eval_folder_path, eval_file_name)
                    highest_coco_values, model_ckpt = get_highest_coco_values(eval_file_path, highest_coco_values, model_ckpt)
            
            if (convert_model):
                export_inference_graph.FLAGS.input_type = "image_tensor"
                export_inference_graph.FLAGS.pipeline_config_path = os.path.join(config.LOG_FOLDER, folder_name, "pipeline.config")
                export_inference_graph.FLAGS.trained_checkpoint_prefix = os.path.join(config.LOG_FOLDER, folder_name, "model.ckpt-" + str(model_ckpt))
                export_inference_graph.FLAGS.output_directory = os.path.join(config.LOG_FOLDER, folder_name, "inference")
                
                image_graph = tf.Graph()
                with image_graph.as_default():
                    export_inference_graph.main([])
            
        if (not highest_coco_values is None):
            for dataset_name in config.DATASET_NAMES:
                parts = folder_name.split("_run_%s_stride8_" % dataset_name) 
            
                if (len(parts) > 1):
                    results_for_scales = aggregated_results.setdefault(dataset_name, {})
                    scales_string = parts[1]
                    
                    results_array = results_for_scales.setdefault(scales_string, [])
                    results_array.append(highest_coco_values)

    for config_key, results_for_scales in aggregated_results.items():
        print (config_key)
        
        for scales_string, results_array in results_for_scales.items():
            scales_comma = ", ".join(scales_string.split("_"))
            
            # remove highest and lowest result
            results_array.sort(key=lambda array: array[0])
            filtered_results_array = results_array[1:-1]
            
            """
            # for printing individual results
            print ("%s\t" % scales_comma)
            
            for result_array in results_array:
                result_array_string = "\t".join(map(lambda result: str(round(result * 100, 2)) + "%", result_array))
                print (result_array_string)
            """
            
            average_results = np.mean(np.array(filtered_results_array), axis=0).tolist()
            average_results = average_results[0:6]
            average_results_string = "\t".join(map(lambda result: str(round(result * 100, 2)) + "%", average_results))
            
            print ("%s\t%s" % (scales_comma, average_results_string))
            
if __name__ == '__main__':
    main()