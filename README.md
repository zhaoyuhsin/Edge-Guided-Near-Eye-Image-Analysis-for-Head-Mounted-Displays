# Edge-Guided Near-Eye Image Analysis for Head Mounted Displays
This is the pytorch implementation of our paper at ISMAR 2021:
> [Edge-Guided Near-Eye Image Analysis for Head Mounted Displays](https://arxiv.org/abs/2009)
>
> Zhimin Wang, Yuxin Zhao, Yunfei Liu, Feng Lu.



## Introduction

Eye tracking provides an effective way for interaction in Augmented Reality (AR) Head Mounted Displays (HMDs). Current eye tracking techniques for AR HMDs require eye segmentation and ellipse fitting under near-infrared illumination. However, due to the low contrast between sclera and iris regions and unpredictable reflections, it is still challenging to accomplish accurate iris/pupil segmentation and the corresponding ellipse fitting tasks. In this paper, inspired by the fact that most essential information is encoded in the edge areas, we propose a novel near-eye image analysis method with edge maps as guidance. Specifically, we first utilize an Edge Extraction Network to predict high-quality edge maps, which only contain eyelids and iris/pupil contours without other undesired edges. Then we feed the edge maps into an Edge-Guided Segmentation and Fitting Network (ESF-Net) for accurate segmentation and ellipse fitting. Extensive experimental results demonstrate that our method outperforms current state-of-the-art methods in near-eye image segmentation and ellipse fitting tasks, based on which we present applications of eye tracking with AR HMD.

## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

## Usage

#### Test

We provide the code to inference based on the well-trained model parameters.

```shell
python3 test.py --curObj $1 --path2data $2 --loadfile $3 --setting $4
```

`curObj` is the dataset name we used to test. There are `LPW`, `NVGaze`, `OpenEDS` and `Fuhl` respectively.  We introduce their meaning in our paper.

`path2data` is the preprocess dataset folder location. 

`loadfile` is the pretrained model file location.

`setting` is  the setting file location. It contains some model settings, such as feature channels, whether to use task-related edge. You can find these setting files in `config` folder.

We release all test log in command terminal.



#### Examples

```shell
python3 test.py --curObj LPW --path2data ../../ --loadfile baseline_edge_16.pkl --setting configs/baseline_edge.yaml
```

#### 
