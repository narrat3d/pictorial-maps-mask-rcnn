import os
import sys

ROOT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIM_FOLDER = os.path.join(ROOT_FOLDER, "slim")
sys.path.append(SLIM_FOLDER)

# where label maps and pipeline configurations are stored
CONFIG_FOLDER = os.path.join(ROOT_FOLDER, "config")

# where checkpoint models are saved
LOG_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Mask-RCNN\logs"

# where training and test images, masks, and keypoints are stored
SOURCE_DATA_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Mask-RCNN\data\character_maps"

# where training and test records are stored
RECORDS_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Mask-RCNN\records"

# where existing models are stored
MODELS_FOLDER = r"C:\Users\sraimund\Pictorial-Maps-Mask-RCNN\models"

LABEL_MAP_NAME = "narrated_label_map.pbtxt"
PIPELINE_NAME = "narrated_pipeline.config"

CROPPED_REGIONS_FOLDER = os.path.join(RECORDS_FOLDER, "cropped_regions")

BEST_MODEL_NAME = "separated_stride8_0.25_0.5_1.0_2.0" 
# internally 3rd_run_separated_new_stride8_0.25_0.5_1.0_2.0

ORIGINAL_MODEL_NAME = "mask_rcnn_resnet101_atrous_coco"
ORIGINAL_COCO_WEIGHTS_PATH = os.path.join(MODELS_FOLDER, ORIGINAL_MODEL_NAME, "model.ckpt")

TRAIN_RECORD_PATH = os.path.join(RECORDS_FOLDER, "%s.record")
TEST_RECORD_PATH = os.path.join(RECORDS_FOLDER, "test.record")

TEST_DATA_PATH = os.path.join(SOURCE_DATA_FOLDER, "test")
COCO_GROUND_TRUTH_PATH = os.path.join(SOURCE_DATA_FOLDER, "test", "coco_ground_truth.json")
PERSON_CATEGORY_ID = 1
PERSON_CATEGORY_NAME = "character"

# optionally comment items in the list 
RECORD_NAMES = [
    # "synthetic",
    # "real",
    "separated",
    # "mixed",
    # "separated_mixed",
    # "test"
]

# optionally comment items in the list 
DATASET_NAMES = [
    # "real",
    # "synthetic",
    "separated",
    # "mixed",
    # "separated_mixed",
]

# optionally comment items in the list 
SCALE_ARRAYS = [
    [0.25, 0.5, 1.0, 2.0],
    # [0.125, 0.25, 0.5, 1.0],
    # [0.0625, 0.125, 0.25, 0.5]
]

STRIDES = [8]
# optionally comment items in the list 
RUN_NRS = [
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th"
]  

EVAL_STEPS = [2304]

INFERENCE_THRESHOLD = 0.3
INFERENCE_FOLDER_NAME = "inference-%s"
INFERENCE_GRAPH_NAME = "frozen_inference_graph.pb"

get_train_record_path = lambda data_set_name: TRAIN_RECORD_PATH % data_set_name
get_test_record_path = lambda: TEST_RECORD_PATH

get_inference_model_path = lambda checkpoint_folder_name, step: os.path.join(LOG_FOLDER, checkpoint_folder_name, INFERENCE_FOLDER_NAME % step, INFERENCE_GRAPH_NAME)

def get_checkpoint_folder_name(run_nr, data_set_name, stride, scales):
    scales_underscore = "_".join(map(str, scales))
    folder_name = "%s_run_%s_stride%s_%s" % (run_nr, data_set_name, stride, scales_underscore)
    
    return folder_name

def get_config_file_template():
    label_map_path = os.path.join(CONFIG_FOLDER, LABEL_MAP_NAME)
    pipeline_template_name = PIPELINE_NAME.replace(".config", "_template.config")
    pipeline_template_path = os.path.join(CONFIG_FOLDER, pipeline_template_name)
    
    config_file_template = open(pipeline_template_path).read()
    config_file_template = config_file_template.replace("$label_map_path$", label_map_path)
    config_file_template = config_file_template.replace("$coco_weights_path$", ORIGINAL_COCO_WEIGHTS_PATH)
    
    return config_file_template


def get_config_file_path():
    return os.path.join(CONFIG_FOLDER, PIPELINE_NAME)


def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)