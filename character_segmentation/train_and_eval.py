import os
from multiprocessing import Process
from character_segmentation import config
from object_detection.legacy import train, eval
import shutil
from character_segmentation.evaluation import filter_coco_results,\
    convert_image_file_names_to_ids, print_coco_results

def mask_rcnn_train(folder_name, config_path):
    print("Start training %s" % folder_name)

    train.FLAGS.train_dir = os.path.join(config.LOG_FOLDER, folder_name)
    train.FLAGS.pipeline_config_path = config_path
    
    train.main(["--logtostderr"])


def mask_rcnn_eval(folder_name, config_path):
    print("Start evaluation %s" % folder_name)

    eval.FLAGS.pipeline_config_path = config_path
    eval.FLAGS.checkpoint_dir = os.path.join(config.LOG_FOLDER, folder_name)
    eval.FLAGS.eval_dir = os.path.join(config.LOG_FOLDER, folder_name, "eval")

    eval.main(["--logtostderr"])


# retrain the original model
def train_and_eval():
    config_file_template = config.get_config_file_template()
    config_file_path = config.get_config_file_path()
    
    for data_set_name in config.DATASET_NAMES:
        for run_nr in config.RUN_NRS:
            for stride in config.STRIDES:
                for scales in config.SCALE_ARRAYS:
                    scales_comma = ", ".join(map(str, scales))
                    
                    train_record_path = config.get_train_record_path(data_set_name)
                    test_record_path = config.get_test_record_path()
                    
                    config_file_content = config_file_template.replace("$stride$", str(stride))
                    config_file_content = config_file_content.replace("$scales$", scales_comma)
                    config_file_content = config_file_content.replace("$train_record_path$", train_record_path)
                    config_file_content = config_file_content.replace("$eval_record_path$", test_record_path)
                    config_file_content = config_file_content.replace("\\", "/")

                    folder_name = config.get_checkpoint_folder_name(run_nr, data_set_name, stride, scales)
                       
                    for current_step in config.EVAL_STEPS:    
                        config_file_content_with_steps = config_file_content.replace("$steps$", str(current_step))
                        
                        with open(config_file_path, "w") as config_file:
                            config_file.write(config_file_content_with_steps)
                        
                        train_process = Process(target=mask_rcnn_train, args=(folder_name, config_file_path))
                        train_process.start()
                        train_process.join()
                    
                        eval_process = Process(target=mask_rcnn_eval, args=(folder_name, config_file_path))
                        eval_process.start()
                        eval_process.join()


def eval_original_model():
    folder_name = config.ORIGINAL_MODEL_NAME
    original_model_path = os.path.join(config.MODELS_FOLDER, folder_name)
    log_folder_path = os.path.join(config.LOG_FOLDER, folder_name)
    
    config.LABEL_MAP_NAME = "mscoco_label_map.pbtxt"
    config.PIPELINE_NAME = "mscoco_pipeline.config"
    
    shutil.copy(os.path.join(original_model_path, "model.ckpt.data-00000-of-00001"), log_folder_path)
    shutil.copy(os.path.join(original_model_path, "model.ckpt.index"), log_folder_path)
    shutil.copy(os.path.join(original_model_path, "model.ckpt.meta"), log_folder_path)
    
    config_file_path = config.get_config_file_path()
    coco_results_path = os.path.join(log_folder_path, "coco_results.json")
    
    config_file_template = config.get_config_file_template()
    config_file_content = config_file_template.replace("$coco_results_path$", coco_results_path)
    config_file_content = config_file_content.replace("$eval_record_path$", config.TEST_RECORD_PATH)
    config_file_content = config_file_content.replace("\\", "/")
    
    with open(config_file_path, "w") as config_file:
        config_file.write(config_file_content)
    
    checkpoint_file_path = os.path.join(log_folder_path, "checkpoint")

    checkpoint_content = "model_checkpoint_path: \"%s\"\n" % os.path.join(log_folder_path, "model.ckpt")
    checkpoint_content += "all_model_checkpoint_paths: \"%s\"" % os.path.join(log_folder_path, "model.ckpt")
    checkpoint_content = checkpoint_content.replace("\\", "\\\\")

    with open(checkpoint_file_path, "w") as checkpoint_file:
        checkpoint_file.write(checkpoint_content)
    
    eval_process = Process(target=mask_rcnn_eval, args=(folder_name, config_file_path))
    eval_process.start()
    eval_process.join()
    
    coco_filtered_results_path = os.path.join(log_folder_path, "filtered_coco_results.json")
    filter_coco_results(coco_results_path, coco_filtered_results_path, [config.PERSON_CATEGORY_ID])
    image_ids = convert_image_file_names_to_ids(coco_filtered_results_path)
    print_coco_results(coco_filtered_results_path, image_ids)


if __name__ == '__main__':
    # eval_original_model()   
    train_and_eval()