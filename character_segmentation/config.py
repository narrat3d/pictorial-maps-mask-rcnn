import os
import sys

slim_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "slim")
sys.path.append(slim_folder)

# where checkpoint models are saved
LOG_FOLDER = r"E:\CNN\logs\mask_rcnn\character_tensorflow"

# where training data is stored
DATA_FOLDER = r"E:\CNN\masks\tensorflow\characters"

SOURCE_DATA_FOLDER = r"E:\CNN\masks\data\character_maps"

# model which will be retrained
COCO_WEIGHTS_PATH = r"E:\CNN\models\mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28\model.ckpt"

CONFIG_FILE_PATH = os.path.join(DATA_FOLDER, "mask_rcnn_inception_resnet_v2_atrous_coco.config")
CONFIG_FILE_TEMPLATE_PATH = os.path.join(DATA_FOLDER, "mask_rcnn_inception_resnet_v2_atrous_coco_template.config")

LABEL_MAP_PATH = os.path.join(DATA_FOLDER, "label_map.pbtxt")
TRAIN_RECORD_PATH = os.path.join(DATA_FOLDER, "train_%s.record")
TEST_RECORD_PATH = os.path.join(DATA_FOLDER, "test.record")

TEST_DATA_PATH = os.path.join(SOURCE_DATA_FOLDER, "test")
COCO_GROUND_TRUTH_PATH = os.path.join(SOURCE_DATA_FOLDER, "test", "coco_ground_truth.json")
PERSON_CATEGORY_ID = 1

DATA_SET_NAMES = [
    "separated",
    # "mixed",
    # "separated_mixed"
]

SCALE_ARRAYS = [
    [0.25, 0.5, 1.0, 2.0],
    # [0.125, 0.25, 0.5, 1.0],
    # [0.0625, 0.125, 0.25, 0.5]
]

STRIDES = [8]
RUN_NRS = ["1st"] 

EPOCHS = 1
STEP_SIZE = 2 * 576 # 576
MAX_STEPS = EPOCHS * STEP_SIZE

INFERENCE_THRESHOLD = 0.6
INFERENCE_FOLDER_NAME = "inference-%s"

get_train_record_path = lambda data_set_name: TRAIN_RECORD_PATH % data_set_name
get_test_record_path = lambda: TEST_RECORD_PATH

get_original_inference_model_path = lambda model_name: r"E:\CNN\models\%s\frozen_inference_graph.pb" % model_name
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