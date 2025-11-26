#!/bin/bash
# bash /media/wyh/star/scripts/voc/overlapped/train_voc_15-5.sh
# GPU=0
# BS=24
# SAVEDIR='saved_voc'

TASKSETTING='overlap'
# TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
# 如需开启 BCE 蒸馏，直接在命令末尾追加： --enable_mbce_distill [--distill_bg_only]
# --mem_size 50
# --epochs 60

NAME='STAR'
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${INIT_LR} \
#--task_name '15-5' --task_step 0 \
#--bs 24 --epochs 80 --val_every 2
#
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 80 --val_every 2








#--------------24ing
####原生代码60次迭代效果有没有保持原样
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${INIT_LR} \
#--task_name '15-5' --task_step 0 \
#--bs 24 --epochs 60 --val_every 10 --info 'train0-bs24-epochs60'
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 60 \
#--val_every 10 --info 'train1-bs24-epochs60'
#
###原生60次迭代加上软标签和过滤背景软标签有没有变好
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill \
#--val_every 2 --info 'train1-bs24-epochs60-enable_mbce_distill'
##
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --distill_bg_only \
#--val_every 2 --info 'train1-bs24-epochs60-distill_bg_only'








