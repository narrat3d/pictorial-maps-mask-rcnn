'''
input: 
model checkpoints

output: 
frozen inference graphs

purpose: 
needs to be done before inferring masks and bounding boxes
'''
import os
from character_segmentation import config
import tensorflow as tf
from object_detection import export_inference_graph
import shutil

for current_step in config.EVAL_STEPS:    
    for data_set_name in config.DATA_SET_NAMES:
        for run_nr in config.RUN_NRS:
            for stride in config.STRIDES:
                for scales in config.SCALE_ARRAYS:
                    folder_name = config.get_checkpoint_folder_name(run_nr, data_set_name, stride, scales)
                    folder_path = os.path.join(config.LOG_FOLDER, folder_name)
                    
                    output_folder = os.path.join(folder_path, (config.INFERENCE_FOLDER_NAME) % current_step)
                    
                    if (os.path.exists(output_folder)):
                        shutil.rmtree(output_folder)
        
                    export_inference_graph.FLAGS.input_type = "image_tensor"
                    export_inference_graph.FLAGS.pipeline_config_path = os.path.join(folder_path, "pipeline.config")
                    export_inference_graph.FLAGS.trained_checkpoint_prefix = os.path.join(folder_path, "model.ckpt-" + str(current_step))
                    export_inference_graph.FLAGS.output_directory = output_folder
                    export_inference_graph.FLAGS.write_inference_graph = True
                    
                    image_graph = tf.Graph()
                    with image_graph.as_default():
                        export_inference_graph.main([])
                    
                    file_names = os.listdir(output_folder)
                    
                    """
                    for file_name in file_names:
                        if (file_name != "frozen_inference_graph.pb"):
                            file_path = os.path.join(output_folder, file_name)
                            
                            if (os.path.isfile(file_path)):
                                os.remove(file_path)
                            else:
                                shutil.rmtree(file_path)
                    """