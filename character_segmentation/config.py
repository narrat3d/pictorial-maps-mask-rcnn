import os
import sys

slim_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slim")
sys.path.append(slim_folder)

# where checkpoint models are saved
LOG_FOLDER = r"E:\CNN\logs\mask_rcnn\character_tensorflow"

# where training images, masks, and keypoints are stored
SOURCE_DATA_FOLDER = r"E:\CNN\masks\data\character_maps"

# where training records are stored
DATA_FOLDER = r"E:\CNN\masks\tensorflow\characters"
CROPPED_IMAGES_FOLDER = os.path.join(DATA_FOLDER, "cropped_images")

# model which will be retrained
MODEL_NAME = "mask_rcnn_resnet101_atrous_coco"
COCO_WEIGHTS_PATH = r"E:\CNN\models\%s_2018_01_28\model.ckpt" % MODEL_NAME
ORIGINAL_INFERENCE_MODEL_PATH = r"E:\CNN\models\%s_2018_01_28\frozen_inference_graph.pb" % MODEL_NAME

CONFIG_FILE_PATH = os.path.join(DATA_FOLDER, "%s.config" % MODEL_NAME)
CONFIG_FILE_TEMPLATE_PATH = os.path.join(DATA_FOLDER, "%s_template.config" % MODEL_NAME)

LABEL_MAP_PATH = os.path.join(DATA_FOLDER, "label_map.pbtxt")
TRAIN_RECORD_PATH = os.path.join(DATA_FOLDER, "train_%s.record")
TEST_RECORD_PATH = os.path.join(DATA_FOLDER, "test.record")

DATASET_NAMES = ["real", "synthetic", "separated", "mixed", "separated_mixed", "test"]

TEST_DATA_PATH = os.path.join(SOURCE_DATA_FOLDER, "test")
COCO_GROUND_TRUTH_PATH = os.path.join(SOURCE_DATA_FOLDER, "test", "coco_ground_truth.json")
PERSON_CATEGORY_ID = 1
PERSON_CATEGORY_NAME = "character"

DATA_SET_NAMES = [
    # "real",
    # "synthetic",
    "separated",
    # "mixed",
    # "separated_mixed",
]

SCALE_ARRAYS = [
    # [0.25, 0.5, 1.0, 2.0],
    # [0.125, 0.25, 0.5, 1.0]
    [0.06125, 0.125, 0.25, 0.5]
]

STRIDES = [8]
RUN_NRS = ["1st"] # "2nd", "3rd"

EVAL_STEPS = [2304]

INFERENCE_THRESHOLD = 0.5
INFERENCE_FOLDER_NAME = "inference-%s"

get_train_record_path = lambda data_set_name: TRAIN_RECORD_PATH % data_set_name
get_test_record_path = lambda: TEST_RECORD_PATH

get_inference_model_path = lambda checkpoint_folder_name, step: os.path.join(LOG_FOLDER, checkpoint_folder_name, 
                                                              INFERENCE_FOLDER_NAME % step, "frozen_inference_graph.pb")

def get_checkpoint_folder_name(run_nr, data_set_name, stride, scales):
    scales_underscore = "_".join(map(str, scales))
    folder_name = "%s_run_%s_stride%s_%s" % (run_nr, data_set_name, stride, scales_underscore)
    
    return folder_name

def get_config_file_template():
    config_file_template = open(CONFIG_FILE_TEMPLATE_PATH).read()
    config_file_template = config_file_template.replace("$label_map_path$", LABEL_MAP_PATH)
    config_file_template = config_file_template.replace("$coco_weights_path$", COCO_WEIGHTS_PATH)
    
    return config_file_template

def mkdir_if_not_exists(file_path):
    if (not os.path.exists(file_path)):
        os.mkdir(file_path)