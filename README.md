# Instance segmentation of persons on maps with Mask-RCNN

## Installation

* Requires [Python 3.7.x](https://www.python.org/downloads/)
* Requires [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-downloads) and corresponding [cuDNN](https://developer.nvidia.com/rdp/cudnn-download)
* Download [this project](https://gitlab.ethz.ch/sraimund/pictorial-maps-mask-rcnn/-/archive/master/pictorial-maps-mask-rcnn-master.zip)
* pip install -r requirements.txt


## Training

* Download [training data](https://ikgftp.ethz.ch/?u=FSVj&p=chOj&path=/pictorial_maps_mask_rcnn_data.zip) and adjust DATA_FOLDER in config.py 
* Set LOG_FOLDER in config.py where intermediate snapshots shall be stored
* Download [trained coco weights](http://download.tensorflow.org/models/object_detection/mask_rcnn_resnet101_atrous_coco_2018_01_28.tar.gz) for Mask-RCNN and set COCO_WEIGHTS_PATH to the downloaded model in config.py
* Optionally adjust properties like data set names (e.g. separated), number of runs (e.g. 1st), scales (e.g. [0.25, 0.5, 1.0, 2.0]), eval steps (e.g. [2304, 4608]) in config.py
* Run train_and_eval.py to train and validate the network in alternating epochs
* Run coco_metrics.py to see individual or average COCO scores from validation checkpoints

## Inference

* first convert your models from saved checkpoint into inference graphs with convert_model.py
* use evaluation.py to draw bounding boxes around persons and transparent masks over persons on the map
* use extract_characters_for_animation.py to mask out persons on the map and store them on separate images
* use extract_characters_for_training.py to store persons on separate images with some or entire background, plus the corresponding keypoints and masks from the [training data](https://ikgftp.ethz.ch/?u=oMwO&p=lzet&path=/persons_on_maps_training_data.zip)


## Sources
* https://github.com/tensorflow/models/tree/master/research/object_detection (Apache License, Copyright by The TensorFlow Authors)
* https://github.com/tensorflow/models/tree/master/research/slim (Apache License, Copyright by The TensorFlow Authors)


## Modifications
* Bug fixes of TensorFlow code based on GitHub issues so that it actually works for Python 3 on Windows
* Commented duplicate FLAGS in eval.py so that it can be used together with train.py 
* JSON dumps of COCO groundtruth and results in coco_evaluation.py
* added area to export_dict for masks in coco_tools.py so that bbox metrics can be also calculated for instance segmentation
* Compiled .proto to .py files
* Parametrization of config files so that it can be trained in multiples runs with different configurations

© 2019-2020 ETH Zurich, Raimund Schnürer

