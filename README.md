# Edge-Guided Near-Eye Image Analysis for Head Mounted Displays
This is the pytorch implementation of our paper at ISMAR 2021:
> [Edge-Guided Near-Eye Image Analysis for Head Mounted Displays](https://arxiv.org/abs/2009)
>
> Zhimin Wang, Yuxin Zhao, Yunfei Liu, Feng Lu.

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

`loadfile` is our pretrained model file location.

`setting` is  the setting file location. It contains some model settings, such as feature channels, whether to use task-related edge. You can find these setting files in `config` folder.

We release all test log in command terminal.



#### Examples

```shell
python3 test.py --curObj LPW --path2data ../../ --loadfile logs/baseline_edge_16.pkl --setting configs/baseline_edge.yaml
```

#### 