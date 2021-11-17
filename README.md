# Edge-Guided Near-Eye Image Analysis for Head Mounted Displays

Please click [our demo video](https://youtu.be/wV1kkvdW5WE).

![image](https://user-images.githubusercontent.com/24918185/142203527-c9d800e1-1950-4339-b858-4685e0b7cd9d.png)

This is the pytorch implementation of our conference paper at **ISMAR 2021**:
> [Edge-Guided Near-Eye Image Analysis for Head Mounted Displays](https://ieeexplore.ieee.org/document/9583797)
>
> Zhimin Wang, Yuxin Zhao, Yunfei Liu, Feng Lu.

The code of this project is modified on [EllSeg](https://github.com/RSKothari/EllSeg) framework, thank RSKothari for enthusiastically solving our doubts for many times.

## Introduction

Eye tracking provides an effective way for interaction in Augmented Reality (AR) Head Mounted Displays (HMDs). Current eye tracking techniques for AR HMDs require eye segmentation and ellipse fitting under near-infrared illumination. However, due to the low contrast between sclera and iris regions and unpredictable reflections, it is still challenging to accomplish accurate iris/pupil segmentation and the corresponding ellipse fitting tasks. In this paper, inspired by the fact that most essential information is encoded in the edge areas, we propose a novel near-eye image analysis method with edge maps as guidance. Specifically, we first utilize an Edge Extraction Network to predict high-quality edge maps, which only contain eyelids and iris/pupil contours without other undesired edges. Then we feed the edge maps into an Edge-Guided Segmentation and Fitting Network (ESF-Net) for accurate segmentation and ellipse fitting. Extensive experimental results demonstrate that our method outperforms current state-of-the-art methods in near-eye image segmentation and ellipse fitting tasks, based on which we present applications of eye tracking with AR HMD.



## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

## Usage

### Prepare data

You can generate datasets files by codes in `dataset_generation` folder, or directly download from [there](https://drive.google.com/drive/folders/1OBsiIhrPxMT_IxVXrTcNBRIwZ6X70YgF?usp=sharing). And put these files in `datasets` folder.
The file structure is as follows:
* datasets
    * TEyeD-mat-Edges
        * *.mat
    * TEyeD-h5-Edges
        * *.h5 

### Train

Comming soon. But we already released our pretrain model, you can try it out on four datasets or your eye videos.


### Test on datasets

We provide the code to inference based on the well-trained model parameters.

```shell
python3 test.py --curObj $1 --path2data $2 --loadfile $3 --setting $4
```

`curObj` is the dataset name we used to test. There are `LPW`, `NVGaze`, `OpenEDS` and `Fuhl` respectively.  We introduce their meaning in our paper.

`path2data` is the preprocess dataset folder location. 

`loadfile` is the pretrained model file location.

`setting` is  the setting file location. It contains some model settings, such as feature channels, whether to use task-related edge. You can find these setting files in `config` folder.

We release all test log in command terminal.

Below is an example：

```shell
python3 test.py --curObj LPW --path2data datasets --loadfile baseline_edge_16.pkl --setting configs/baseline_edge.yaml
```

### Test on your eye videos 
```shell
python3 evaluate.py --path2data $1 
```

`path2data` is the folder location of your eye videos. 
We release the result videos in `path2data`.

Below is an example：

```shell
python3 evaluate.py  --path2data video
```
There is a example video `example1.avi` in `video` folder. It's fit result will automatically placed in the same folder.

# Citations

If you only use our code base, please cite the following works
```
@article{9583797,
     author={Wang, Zhimin and Zhao, Yuxin and Liu, Yunfei and Lu, Feng},
     booktitle={2021 IEEE International Symposium on Mixed and Augmented Reality (ISMAR)}, 
     title={Edge-Guided Near-Eye Image Analysis for Head Mounted Displays}, 
     year={2021},
     volume={},
     number={},
     pages={11-20},
     doi={10.1109/ISMAR52148.2021.00015}
}
```


