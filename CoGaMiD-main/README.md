# Continual Gaussian Mixture Distribution Modeling for Class Incremental Semantic Segmentation

This is an official implementation of the paper "Continual Gaussian Mixture Distribution Modeling for Class Incremental Semantic Segmentation"。

<img src = "https://github.com/zhu-gl-ux/CoGaMiD/blob/main/figures/overview.png" width="100%" height="100%">

# Abstract
Class incremental semantic segmentation (CISS) enables a model to continually segment new classes from non-stationary data while preserving previously learned knowledge. Recent top-performing approaches are prototype-based methods that assign a prototype to each learned class to reproduce previous knowledge. However, modeling each class distribution relying on only a single prototype, which remains fixed throughout the incremental process, presents two key limitations: (i) a single prototype is insufficient to accurately represent the complete class distribution when incoming data stream for a class is naturally multimodal; (ii) the features of old classes may exhibit anisotropy during the incremental process, preventing fixed prototypes from faithfully reproducing the matched distribution. To address the aforementioned limitations, we propose a Continual Gaussian Mixture Distribution(CoGaMiD) modeling method. Specifically, the means and covariance matrices of the Gaussian Mixture Models (GMMs) are estimated to model the complete feature distributions of learned classes. These GMMs are stored to generate pseudo features that support the learning of novel classes in incremental steps. Moreover, we introduce a Dynamic Adjustment (DA) strategy that utilizes the features of previous classes within incoming data streams to update the stored GMMs. This adaptive update mitigates the mismatch between fixed GMMs and continually evolving distributions. Furthermore, a Gaussian-based Representation Constraint (GRC) loss is proposed to enhance the discriminability of new classes, avoiding confusion between new and old classes. Extensive experiments on Pascal VOC and ADE20K show that our method achieves superior performance compared to previous methods, especially in more challenging long-term incremental scenarios.

# Getting Started

### Requirements
- python==3.11.4
- torch==1.12.1
- torchvision==0.13.1
- numpy
- pillow
- scikit-learn
- tqdm
- matplotlib


### Datasets
```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationClassAug/
    --- ADEChallengeData2016
        --- annotations
            --- training
            --- validation
        --- images
            --- training
            --- validation
```
You can download the original dataset in [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit). To train our model with augmented samples, please download labels of augmented samples (['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip)) and file names (['train_aug.txt'](https://github.com/cvlab-yonsei/DKD/releases/download/v1.0/train_aug.txt)). For ADE20K, you can dawnload the dataset in [here](http://sceneparsing.csail.mit.edu/).

### Class-Incremental Segmentation Segmentation on VOC 2012

```Shell

GPU=0
BS=24 
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='1-1'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 



NAME='CoGaMiD'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```

### Class-Incremental Segmentation Segmentation on ADE20K

```shell
GPU=0,1
BS=12  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='100-50'
INIT_LR=0.0025
LR=0.00025
MEMORY_SIZE=0

NAME='CoGaMiD'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
```


### Testing

```Shell
python eval_voc.py --device 0 --test --resume path/to/weight.pth
```



## Citation
```
@inproceedings{zhucontinual,
  title={Continual Gaussian Mixture Distribution Modeling for Class Incremental Semantic Segmentation},
  author={Zhu, Guilin and Wang, Runmin and Shao, Yuanjie and dong Yang, Wei and Sang, Nong and Gao, Changxin},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

## Acknowledgements
* This code is based on [DKD](https://github.com/cvlab-yonsei/DKD) ([2022-NeurIPS]) and [STAR](https://github.com/jinpeng0528/STAR) [2023-NeurIPS].
* This template is borrowed from [pytorch-template](https://github.com/victoresque/pytorch-template).
