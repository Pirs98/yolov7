# Investigation on the Effectiveness of Tools for Optimizing YOLOv7 Training Performance

Implementation of Pietro Santandreas's Master Thesis - [Investigation on the Effectiveness of Tools for Optimizing Object Detector Training Performance](https://drive.google.com/file/d/1vU1hcDD94EkHLp1kBoUp6L8vFKTtT03F/view?usp=sharing)

## Overview

This repository contains the implementation of the Thesis mentioned above. This work was carried out at the DLR (German Aerospace Center) and focused on tools for optimizing training performance. The model used was YOLOv7 ([link here](https://github.com/WongKinYiu/yolov7)), and the tools in focus were Data Augmentation, Dropout Regularization, and Hard Examples Mining. The raw results can be seen [here](scripts/results.xlsx).

## Results Table

| Model | Description | P-Bike | P-Car | P-Lorry | P-Bus | P-Pedestrian | mAP@.5 | mAP@.5:.95 |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
**M0** | No Augmentations | 0.555 | 0.908 | 0.768 | 0.740 | 0.278 | 0.652 | 0.396 |
**M1** | Spatial | 0.675 | 0.923 | 0.800 | 0.790 | 0.375 | 0.677 | 0.410 |
**M2** | Color | 0.563 | 0.900 | 0.755 | 0.575 | 0.290 | 0.627 | 0.373 |
**M3** | Spatial<br>Color | 0.643 | 0.915 | 0.785 | 0.715 | 0.315 | 0.655 | 0.392 |
**M4** | Spatial<br>Color<br>Mosaic | 0.740 | 0.940 | 0.885 | 0.860 | 0.460 | 0.762 | 0.475 |
**M4.1** | Spatial<br>Mosaic | 0.735 | 0.940 | 0.890 | 0.888 | 0.473 | 0.767 | 0.475 |
**M5** | M4.1 setup<br>Dropout | 0.710 | 0.935 | 0.850 | 0.775 | 0.408 | 0.732 | 0.446 |
**M6** | M5 setup<br>+100 Hard Samples | 0.748 | 0.950 | 0.888 | 0.883 | 0.558 | 0.790 | 0.492 |
**M6.1** | M4.1 setup<br>+100 Hard Samples | 0.773 | 0.950 | 0.893 | 0.858 | 0.598 | **0.803** | **0.507** |
**M7** | M5 setup<br>+100 Random Samples | 0.708 | 0.948 | 0.885 | 0.860 | 0.460 | 0.770 | 0.488 |
    
## Other Trained Models

<details>
<summary>Click to expand</summary>

| Model | Description | P-Bike | P-Car | P-Lorry | P-Bus | P-Pedestrian | mAP@.5 | mAP@.5:.95 |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
**M4.2** | Mosaic | 0.710 | 0.930 | 0.870 | 0.890 | 0.460 | 0.747 | 0.460 |
**M4.3** | M4.1 Setup<br>Mixup | 0.750 | 0.940 | 0.870 | 0.840 | 0.480 | 0.768 | 0.474 |
**M5.1** | M4.1 Setup<br>DropBlock  | 0.788 | 0.950 | 0.905 | 0.885 | 0.540 | 0.798 | 0.510 |
**M6.2** | M4.1 Setup<br>+100 Max Hard Samples | 0.643 | 0.915 | 0.785 | 0.715 | 0.315 | 0.655 | 0.392 |
**M6.3** | M4.1 Setup<br>+200 Hard Samples | 0.798 | 0.953 | 0.890 | 0.890 | 0.620 | 0.809 | 0.509 |
**M6.4** | M4.1 Setup<br>+100 Entropy Images | 0.780 | 0.950 | 0.878 | 0.883 | 0.593 | 0.803 | 0.507 |
**M8** | M4.1 setup<br>All images | 0.870 | 0.960 | 0.920 | 0.890 | 0.690 | 0.878 | 0.582 |
</details>

# New inference scripts

The new Python files added to the previous version are:

* `detect_entropy.py`: computes entropy as the average of the one of each object and plots the result within the image.
* `detect_entropy_list.py`: computes entropy as the average of the one of each object and creates a list sorted by values. The file list path must be written [here](https://github.com/Pirs98/yolov7/blob/main/detect_entropy_list.py#L139).
* `detect_uncertainty.py`: computes uncertainty using the Monte Carlo Dropout Method and plots the result within the image. Dropout architecture is required for correct operation.
* `detect_uncertainty_list.py`: computes uncertainty using the Monte Carlo Dropout Method as the average of the one of each object and creates a list sorted by values. Dropout architecture is required for correct operation. The file list path must be written [here](https://github.com/Pirs98/yolov7/blob/main/detect_uncertainty_list.py#L250C5-L250C24).
* `detect_uncertainty_list_max.py`: computes uncertainty using the Monte Carlo Dropout Method as the maximum uncertainty among all objects and creates a list sorted by values. Dropout architecture is required for correct operation. The file list path must be written [here](https://github.com/Pirs98/yolov7/blob/main/detect_uncertainty_list_max.py#L250).

# Configuration files

The `yaml` configuration files can be split into:
* The `data` file contains paths to datasets and class names.
* The `cfg` file represents the architecture of the model.
* The `hyp` file provides a way to set hyperparameters.

New files are:
* [`custom.yaml`](data/custom.yaml) et al. with the paths and classes of the dataset.
* [`yolov7_dropout.yaml`](cfg/training/yolov7_dropout.yaml) and [`yolov7_dropblock.yaml`](cfg/training/yolov7_dropblock.yaml), which implement drop layers within yolo architectures.
* [`data/hyp.scratch.*.yaml`](data) files implement the Data Augmentation configuration to reach the results shown in the table above.

Docker environment (recommended)

## Training

Be sure to have a correct dataset for training YOLOv7. Pretrained weights are taken from [`yolov7_training.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt).

``` shell
# train M0
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 1280 1280 --epochs 150 --cfg cfg/training/yolov7.yaml --weights '/storage/yolov7_training.pt' --name yolov7_no_aug --hyp data/hyp.scratch.custom_no_aug.yaml

# train M6.1
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom_100_HM.yaml --img 1280 1280 --epochs 150 --cfg cfg/training/yolov7.yaml --weights '/storage/yolov7_training.pt' --name yolov7_best_model --hyp data/hyp.scratch.custom_all_aug_no_color.yaml
```

## Inference for Uncertainty Quantification

Note that the Monte Carlo Dropout Method can be heavily time-consuming.

```
# create list of hard samples
python detect_uncertainty_list.py --weights /runs/training/5_all_aug_no_color_drop.pt --conf 0.4 --img-size 1280 --source /path/to/dir
```

## Other Scripts

I created some scripts for managing files.

* [`compare_files.py`](scripts/compare_files.py) compares the first 100 images from two uncertainty list files and returns the number of images with the same path.
* [`convert_xml_to_yolo.py`](scripts/convert_xml_to_yolo.py) is the file I used to convert XML annotation file to yolo and to create the final dataset. The input [path](https://github.com/Pirs98/yolov7/blob/main/scripts/convert_xml_to_yolo.py#L43) is a directory that contains other repositories with JPEP or PNG images and XML labels, and the [output](https://github.com/Pirs98/yolov7/blob/main/scripts/convert_xml_to_yolo.py#L44) path indicates where to create the new yolo-like dataset.
* [`create_HM_dataset.py`](scripts/create_HM_dataset.py) uses the uncertainty text file to create a new yolo-like dataset. Other input parameters are `N`, which is the number of images, and the optional parameter `--random`, which forces the program to select random images instead of the most uncertain.
* [`create_small_dataset.py`](scripts/create_small_dataset.py) creates a smaller version of the given dataset according to the `ratio` value.
* [`split_dataset.py`](scripts/split_dataset.py) splits a dataset into train, validation, and test sets according to the values of the ratios.
