# Official Code for SOFW
This is the official code for our paper "SOFW: A Synergistic Optimization Framework for Indoor 3D Object Detection"
![comparison](https://github.com/mooncake199809/SOFW/blob/main/docs/overall.png)

In this work, we propose SOFW, a synergistic optimization framework that investigates the feasibility of optimizing 3D object detection tasks concurrently spanning several dataset domains.
The core of SOFW is identifying domain-shared parameters to encode universal scene attributes, while employing domain-specific parameters to delve into the particularities of each scene domain. 


# Main Results
In this work, we conduct 3D object detection task on S3DIS, ScanNet, and SUN RGB-D dataset.
The detection accuracy on these datasets is shown as follow.

## S3DIS 
 Method       | VoteNet (baseline) |  SOFW  |  SOFW+  | 
--------------|--------------------|--------|---------|
 mAP@0.25     |         52.0       | 57.48  |  67.69  |
 mAP@0.5      |         23.1       | 33.74  |  40.35  |

## ScanNet V2
 Method       | VoteNet (baseline) |  SOFW  |  SOFW+  | 
--------------|--------------------|--------|---------|
 mAP@0.25     |         62.3       | 68.31  |  70.89  |
 mAP@0.5      |         40.8       | 48.95  |  52.31  |

 ## SUN RGB-D 
 Method       | VoteNet (baseline) |  SOFW  |  SOFW+  | 
--------------|--------------------|--------|---------|
 mAP@0.25     |         59.8       | 62.62  |  65.20  |
 mAP@0.5      |         35.8       | 40.54  |  41.95  |

# Create Environment
SOFW is built upon MMdetection3d (https://github.com/open-mmlab/mmdetection3d). Please follow MMdetection3d to create the environment and process datasets.

# Evaluation
The pre-trained models for SOFW and SOFW+ on the S3DIS, ScanNet, and SUN RGB-D dataset can be downloaded from (https://drive.google.com/drive/folders/1r6DJCKma7PhJrsLxJpkuQ4MH9nHR7BPc).

## Evaluation on the S3DIS Dataset
We can simply run the following code to conduct evaluation on the S3DIS dataset.
```
bash tools/dist_test.sh ./configs/votenet/votenet_8x8_s3dis-3d-5class.py ./weights/SOFW_S3DIS.pth 1 --eval bbox
```
When evaluating SOFW+, please utilize the commented code in ./configs/_base_/models/votenet.py to modified the architecture of the backbone.

## Evaluation on the ScanNet Dataset
We can simply run the following code to conduct evaluation on the ScanNet dataset.
```
bash tools/dist_test.sh ./configs/votenet/votenet_8x8_scannet-3d-18class.py ./weights/SOFW_SCANNET.pth 1 --eval bbox
```

## Evaluation on the SUN RGB-D Dataset
We can simply run the following code to conduct evaluation on the SUN RGB-D dataset.
```
bash tools/dist_test.sh ./configs/votenet/votenet_16x8_sunrgbd-3d-10class.py ./weights/SOFW_SUNRGBD.pth 1 --eval bbox
```