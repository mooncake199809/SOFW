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
 mAP@0.25     | 52.0               | 57.48  | 67.69   |
 mAP@0.5      | 23.1               | 33.74  | 40.35   |

## ScanNet V2
 Method       | VoteNet (baseline) |  SOFW  |  SOFW+  | 
--------------|--------------------|--------|---------|
 mAP@0.25     | 62.3               | 68.31  | 70.89   |
 mAP@0.5      | 40.8               | 48.95  | 52.31   |

 ## SUN RGB-D 
 Method       | VoteNet (baseline) |  SOFW  |  SOFW+  | 
--------------|--------------------|--------|---------|
 mAP@0.25     | 59.8               | 62.62  | 65.20   |
 mAP@0.5      | 35.8               | 40.54  | 41.95   |
