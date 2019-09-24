# Deep High-Resolution Representation Learning for Human Pose Estimation

## Introduction

This repo is **[TensorFlow](https://www.tensorflow.org)** implementation of **[Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)](https://arxiv.org/abs/1902.09212)**.

## Dependencies
* [Anaconda](https://www.anaconda.com/download/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [TensorFlow](https://www.tensorflow.org/)
* [COCO API](https://github.com/cocodataset/cocoapi)

Python 3.7.1 version with Anaconda 3 is used for development.

## Directory

### Dataset
You need prepare the data in the `dataset` directory as below.
```
${POSE_ROOT}
|-- dataset
    |-- COCO
    |-- |-- dets
    |   |   |-- human_detection.json
    |   |-- annotations
    |   |   |-- person_keypoints_train2017.json
    |   |   |-- person_keypoints_val2017.json
    |   |   `-- image_info_test-dev2017.json
    |   `-- images
    |       |-- train2017/
    |       |-- val2017/
    |       `-- test2017/
```
* In the training stage, GT human bbox is used, and `human_detection.json` is used in testing stage.
* Prepare `human_detection.json` following [MS COCO format](http://cocodataset.org/#format-results).
* The imagenet pre-trained hrnet models can be downloaded from [here]().

### Output
You need to follow the directory structure of the `output` folder as below.
```
${POSE_ROOT}
|-- output
    |-- log
    |-- model_dump
    `-- result
```
* Creating `output` folder as soft link form is recommended instead of folder form because it would take large storage capacity.
* `log` folder contains training log file.
* `model_dump` folder contains saved checkpoints for each epoch.
* `result` folder contains final estimation files generated in the testing stage.
* You can change default directory structure of `output` by modifying `config.py`.

## Run
### Start
* Install the dependencies.
* Run `pip install -r requirements.txt` to install other required modules.
* Run `python setup.py build_ext --inplace; rm -rf build` in `nms/` to build NMS modules.
* In `config.py`, you can change settings of the model including dataset to use, network backbone, input size and so on.

### Train
* Run `python train.py --gpu 0,1,2,3` to train the network from scratch on GPU 0,1,2,3. 
* If you want to fine-tune from a pre-trained model, run `python train.py --gpu 0,1,2,3 --weights path_to_model`. 

### Test
* Run `python test.py --gpu 0,1,2,3 --test_model path_to_model` to test the network on the GPU 0,1,2,3.

## Results on MSCOCO val2017
