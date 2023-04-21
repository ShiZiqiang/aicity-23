This repos is the implementation of our paper "CheckSORT: Refined Synthetic Data Combination and Optimized SORT for Automatic Retail Checkout", which will be officially published at AICity23 workshop.

It is based on 
https://github.com/w-sugar/DTC_AICITY2022 &&
[mmdetection](https://github.com/open-mmlab/mmdetection) && 
[mmclassification](https://github.com/open-mmlab/mmclassification) && 
[DeepSort](https://github.com/nwojke/deep_sort) && 
[StrongSORT](https://github.com/dyhBUPT/StrongSORT) &&
[retinex] https://github.com/muggledy/retinex) &&
https://github.com/cybercore-co-ltd/Track4_aicity_2022 &&
https://github.com/a-nau/synthetic-dataset-generation/.
Many thanks for their sharing.

## Introduction

In this paper, we propose a method called CheckSORT for automatic retail checkout. We demonstrate CheckSORT on the multi-class product counting and recognition task in Track 4 of AI CITY CHALLENGE 2023. This task aims to count and identify products as they move along a retail checkout white tray, which is challenging due to occlusion, similar appearance, or blur. Based on the constraints and training data provided by the sponsor, we propose two new ideas to solve this task. The first idea is to design a controllable synthetic training data generation paradigm to bridge the gap between training data and real test videos as much as possible. The second innovation is to improve the efficiency of existing SORT tracking algorithms by proposing decomposed Kalman filter and dynamic tracklet feature sequence. Our experiments resulted in state-of-the-art (when compared with DeepSORT and StrongSORT) F1-scores of 70.3\% and 62.1\% on the TestA data of AI CITY CHALLENGE 2022 and 2023 respectively in the estimation of the time (in seconds) for the product to appear on the tray. 

## Quick & Easy Start

### 1. Environments settings

* python 3.7.12
* pytorch 1.10.0
* torchvision 0.11.1
* cuda 11.3
* mmcv-full 1.4.3
* tensorflow-gpu 1.15.0

```shell
1. conda create -n submit python=3.7
2. conda activate submit
3. conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
4. pip install -r requirements.txt
5. pip uninstall setuptools
6. conda install setuptools==58.0.4 
```

Please download the mmdetection and classification related packages from
(https://drive.google.com/file/d/1uV_wzjVYZQoPf2rSmhg1PIq9fQUk0ccu/view?usp=sharing), and place them in the root dir of this repo. The structure is like 
```
repo/
├── mm_install_package
├── mmclassification
├── mmdetection
├── configs
├── deep_sort
├── src
├── strong_sort
├── tools
├── win_sort
├── work_dirs
├── README.md
├── requirements.txt
```


### 2. Testing
Please place the videos you want to test in the [test_videos](./test_videos) folder.

```
test_videos/
├── testA_1.mp4
├── testA_2.mp4
├── testA_3.mp4
├── testA_4.mp4
├── video_id.txt
```

Please place all the models in the [checkpoints](./checkpoints) folder. These checkpoints can be downloaded from (https://drive.google.com/file/d/1IferVoCo1Nk9YdziVMOeuXj9KT1lYJS8/view?usp=sharing).
The structure is like
```
checkpoints/
├── /resnest101/epoch_20.pth
├── /s50/epoch_20.pth
├── /b0/epoch_20.pth
├── /b2/epoch_20.pth
├── /detectors_cascade_rcnn_r50_1x_coco/epoch_5.pth
├── detectors_cascade_rcnn.pth
```

Please place all the third-party publically available pre-trained models in [models](./models) folder. 
These pre-trained models can be downloaded from 
(https://drive.google.com/file/d/1VqCGUOvZqesCUlKlMfg-nlPAS2CSak4I/view?usp=sharing).
The structure is like
```
models/
|---detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth
|---efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth
|---efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth
|---efficientnet-b2_3rdparty_8xb32-aa-advprop_in1k_20220119-1655338a.pth
|---efficientnet-b2_3rdparty_8xb32_in1k_20220119-ea374a30.pth
|---repvgg-B2_3rdparty_4xb64-coslr-120e_in1k_20210909-bd6b937c.pth
|---resnest101_imagenet_converted-032caa52.pth
|---resnest50_imagenet_converted-1ebf0afe.pth
|---swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth
```
All these pre-trained models can be found at https://download.openmmlab.com. For example
https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_cascade_rcnn_r50_1x_coco/detectors_cascade_rcnn_r50_1x_coco-32a10ba0.pth
https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth

Then we can do the prediction with our models. Please use the following command line
### 3. Use the FFmpeg library to extract/count frames.
python tools/extract_frames.py --out_folder ./frames

### 4. Scan all video images to determine the type of product and when it appears on the white tray (in frame level)
CUDA_VISIBLE_DEVICES=0 python tools/test_net_23_inframe.py --input_folder ./frames --out_file ./results.txt --detector ./checkpoints/detectors_cascade_rcnn_r50_1x_coco/epoch_5.pth --feature ./checkpoints/b0/epoch_20.pth --b2 ./checkpoints/b2/epoch_20.pth --resnest50 ./checkpoints/resnest50/epoch_20.pth --resnest101 ./checkpoints/resnest101/epoch_20.pth

The output result is in results.txt, you can submit it to the official evaluation system to obtain the F1-score.

## Training of models

Download images and annotations for training detector and classifiers from 
(https://drive.google.com/file/d/1zhIEYGuDviOr4N5ZV8nNbWcIDSB2a2oY/view?usp=sharing).

Download images and annotations for training detector from 
https://drive.google.com/file/d/1JiDWpVj3PJG-Kv1YyAy8F_gSifviamKu/view?usp=sharing

Download images and annotations for training classifiers from 
https://drive.google.com/file/d/1k3gjYMnTPFt6ieiPUdzYQ9BXQW12T42O/view?usp=sharing

Put all the downloaded data into the [data](./data) folder, whose structure is like
```
data
├── coco_offline_MSRCR_GB_halfbackground_size100_no-ob_1
│   └── instances_train.json
│   └── train2017
├── alladd2
│   └── meta
│   └── train
│   └── val
```

Then we can start to train the classifiers and detector.

### 1. Train Detector with Multi GPUs(e.g. 4 GPUs)
bash ./mmdetection/tools/dist_train.sh ./mmdetection/configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py 4

The trained detectors will be saved to ./mmdetection/work_dirs/detectors_cascade_rcnn

### 2. Train Classifier with Multi GPUs(e.g. 4 GPUs)
bash ./mmclassification/tools/train_multi.sh 4

The trained classifiers will be saved to ./mmclassification/work_dirs/


## Data preparation

The training data for classifers and detectors are obtained through another repo from us. Pleae refer to https://github.com/ShiZiqiang/checkout-data-generation/.


## Contact

If you have any questions, feel free to contact Ziqiang Shi (shiziqiang@fujitsu.com).

## Reference

```
@InProceedings{shi23AIC23,
	author = {Ziqiang Shi and Zhongling Liu and Liu Liu and Rujie Liu and Takuma Yamamoto and Xiaoyu Mi and Daisuke Uchida},
	title = {CheckSORT: Refined Synthetic Data Combination and Optimized SORT for Automatic Retail Checkout},
	booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	month = {June},
	year = {2023},
}
```
