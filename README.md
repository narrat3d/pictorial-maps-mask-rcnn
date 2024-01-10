# Instance segmentation of human figures on pictorial maps with Mask-RCNN

This is code for the article [Instance Segmentation, Body Part Parsing, and Pose Estimation of Human Figures in Pictorial Maps](https://doi.org/10.1080/23729333.2021.1949087). Visit the [project website](http://narrat3d.ethz.ch/segmentation-of-human-figures-in-pictorial-maps/) for more information.

## Installation

* Requires [Python 3.7.x](https://www.python.org/downloads/)
* Requires [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
* Download [this project](https://gitlab.ethz.ch/narrat3d/pictorial-maps-mask-rcnn/-/archive/master/pictorial-maps-mask-rcnn-master.zip)
* pip install -r requirements.txt

## Data preparation
* Adjust the RECORDS_FOLDER in config.py where records are stored
* Download [training data](https://ikgftp.ethz.ch/?u=w7pz&p=zmEX&path=/human_figures_on_maps_training_data.zip) and [test data](https://ikgftp.ethz.ch/?u=zehc&p=QG2Z&path=/human_figures_on_maps_test_data.zip) into the same folder and set SOURCE_DATA_FOLDER in config.py to this folder
* Optionally adjust RECORD_NAMES in config.py to those folder names which shall be converted
* Optionally run create_mask_tf_record.py to convert the source images into TensorFlow records
* Alternatively download already converted [training and test records](https://ikgftp.ethz.ch/?u=Ub6f&p=yFhR&path=/human_figures_on_maps_data.zip) and store them in the RECORDS_FOLDER

## Training
* Set LOG_FOLDER in config.py where intermediate snapshots shall be stored
* Download the original Mask R-CNN model trained on COCO images from [here](https://ikgftp.ethz.ch/?u=qj9J&p=Pc8P&path=/human_figures_on_maps_models.zip) (or [TensorFlow](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz)) and set MODELS_FOLDER to the path in config.py
* Optionally adjust properties like dataset names (e.g. separated), number of runs (e.g. 1st), scales (e.g. [0.25, 0.5, 1.0, 2.0]), eval steps (e.g. [2304]) in config.py
* Run train_and_eval.py to train and validate the network in alternating epochs
* Run coco_metrics.py to see individual or average COCO scores from validation checkpoints

## Evaluation
#### Retrained models
* First convert your models from saved checkpoint into inference graphs with convert_model.py
* Use evaluation.py to draw bounding boxes around persons and transparent masks over persons on the map

#### Existing models
* Download the original and best retrained model from [here](https://ikgftp.ethz.ch/?u=qj9J&p=Pc8P&path=/human_figures_on_maps_models.zip) and set MODELS_FOLDER to the path in config.py
* Run evaluate_best_model() in evaluation.py to evaluate the best frozen model trained with the data above
* Run evaluate_original_model() in evaluation.py to evaluate the original frozen model trained on COCO images
* Run eval_original_model() in train_and_eval.py to evaluate the original checkpoint model trained on COCO images
* Note that there is a small difference between the two COCO scores due to a [TensorFlow bug](https://github.com/tensorflow/models/issues/8624)

## Sources
* https://github.com/tensorflow/models/tree/master/research/object_detection (Apache License, Copyright by The TensorFlow Authors)
* https://github.com/tensorflow/models/tree/master/research/slim (Apache License, Copyright by The TensorFlow Authors)

## Modifications
* Bug fixes of TensorFlow code based on GitHub issues so that it actually works for Python 3 on Windows
* Commented duplicate FLAGS in eval.py so that it can be used together with train.py 
* JSON dumps of COCO groundtruth and results in coco_evaluation.py
* added area to export_dict for masks in coco_tools.py so that bbox metrics can be also calculated for instance segmentation
* only the latest training model snapshot will be saved (max_to_keep=1)
* Compiled .proto to .py files
* Parametrization of config files so that it can be trained in multiples runs with different configurations

© 2019-2021 ETH Zurich, Raimund Schnürer

