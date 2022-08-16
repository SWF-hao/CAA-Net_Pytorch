# CAA-Net

Official Pytorch Code base for CAA-Net

## Introduction

It is designed for automatic diagnosis of infectious keratitis.

## Using the code:

The code is stable while using Python 3.8.8, CUDA >=10.1

- Clone this repository:

```
git clone https://github.com/SWF-hao/CAA-Net_for_IK_diagnosis
cd CAA-Net-Pytorch
```



Environment configuration, install these dependencies using pip:

```
torch==1.8.1
torchvision==0.9.1
```

## Data Format

Make sure to put the files as the following structure:

```
inputs
└── <dataset name>
├── 001.jpg
├── 002.jpg
├── 003.jpg
├── ...
```

## Download Weights

The weight is preserved in BaiduNetdisc

```
链接: https://pan.baidu.com/s/1Fd3SIHX2Wgb4sz5_2xAXuA 提取码: w9pv
```

Then move the weight to the folder "weights"

## Model Inference

1. Model inference on examples.

```
python model_inferencing.py --weight CAA-Net.pt --batch-size 1 --input-images examples --device cpu --output-name examples
```

2. Model inference on customized dataset.

```
python model_inferencing.py --weight <weight name> --batch-size 1 --input-images <data dir name> --device cpu --output-name <output .csv file name>
```

### Acknowledgements:

This code-base uses certain code-blocks and helper functions from [Pytorch](https://github.com/pytorch/pytorch).

### Citation:

Article is still on revision. It will be released soon.# CAA-Net_for_IK_diagnosis
