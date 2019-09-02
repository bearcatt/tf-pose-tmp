# Deep High-Resolution Representation Learning for Human Pose Estimation

## Introduction

This repo is **[TensorFlow](https://www.tensorflow.org)** implementation of **[Deep High-Resolution Representation Learning for Human Pose Estimation (CVPR 2019)](https://arxiv.org/abs/1902.09212)** of MSRA for **2D multi-person pose estimation** from a single RGB image.

## Dependencies
* [Anaconda](https://www.anaconda.com/download/)
* [CUDA](https://developer.nvidia.com/cuda-downloads)
* [cuDNN](https://developer.nvidia.com/cudnn)
* [TensorFlow](https://www.tensorflow.org/)
* [COCO API](https://github.com/cocodataset/cocoapi)

Python 3.7.1 version with Anaconda 3 is used for development.

## Directory

### Root
The `${POSE_ROOT}` is described as below.
```
${POSE_ROOT}
|-- core
|-- data
|-- hrnet
|-- nms
|-- tfflat
|-- tools
```
* `core` contains codes for train engine, test engine, basic dataloader and the pose model.
* `data` contains the codes for data processing on COCO, MPII and PoseTrack.
* `hrnet` contains codes for building HRNets.
* `nms` contains codes for NMS.
* `tfflat` contains useful codes for data parallel, multi-gpu training, model saving and loading.
* `tools` contains codes that convert MPII and PoseTrack data into COCO format.


### Dataset
You need prepare the data in the `dataset` directory as below.
```
${POSE_ROOT}
|-- dataset
    |-- MPII
    |-- |-- dets
    |   |   |-- human_detection.json
    |   |-- annotations
    |   |   |-- train.json
    |   |   `-- test.json
    |   |-- images
    |   |   |-- 000001163.jpg
    |   `-- mpii2coco.py
    |
    |-- PoseTrack
    |-- |-- dets
    |   |   |-- human_detection.json
    |   |-- annotations
    |   |   |-- train2018.json
    |   |   |-- val2018.json
    |   |   `-- test2018.json
    |   |-- original_annotations
    |   |   |-- train/
    |   |   |-- val/
    |   |   `-- test/
    |   |-- images
    |   |   |-- train/
    |   |   |-- val/
    |   |   `-- test/
    |   `-- posetrack2coco.py
    |
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
    |
    `-- imagenet_weights
        |-- hrnet_w30.ckpt
        |-- hrnet_w32.ckpt
        |-- hrnet_w48.ckpt
```
* Creating `dataset` folder (or its sub-folders) as soft link(s) form is recommended instead of folder form because it would take large storage capacity.
* Use the python script in `${POSE_ROOT}/tools/` to convert MPII and PoseTrack annotation files to [MS COCO format](http://cocodataset.org/#format-data).
* If you want to add your own dataset, you have to convert it to [MS COCO format](http://cocodataset.org/#format-data).
* Except for `annotations` of the MPII and PoseTrack, all other directories are original version of downloaded ones.
* In the training stage, GT human bbox is used, and `human_detection.json` is used in testing stage which should be prepared before testing and follow [MS COCO format](http://cocodataset.org/#format-results).
* Download imagenet pre-trained hrnet models from [here]() and place them in the `data/imagenet_weights`.

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
* You can change default directory structure of `output` by modifying `main/config.py`.

## Run
### Start
* Install the dependencies.
* Run `pip install -r requirements.txt` to install other required modules.
* Run `python setup.py build_ext --inplace; rm -rf build` in `${POSE_ROOT}/nms` to build NMS modules.
* In the `${POSE_ROOT}/config.py`, you can change settings of the model including dataset to use, network backbone, input size and so on.

### Train
In `${POSE_ROOT}`, run
```bash
python train.py --gpu 0,1,2,3
```
to train the network on the GPU 0,1,2,3. 

If you want to continue experiment, run 
```bash
python train.py --gpu 0,1,2,3 --continue
```
and the training will restart from the latest checkpoint.

### Test
Place trained model at the `cfg.model_dump_dir` and human detection result (`human_detection.json`) at `dataset/$DATASET/dets/`.

In `${POSE_ROOT}`, run 
```bash
python test.py --gpu 0,1,2,3 --test_epoch 210
```
to test the network on the GPU 0,1,2,3 with 210th epoch trained model.

