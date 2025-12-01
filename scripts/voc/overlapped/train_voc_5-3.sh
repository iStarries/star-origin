#!/bin/bash

# bash /media/wyh/star/scripts/voc/overlapped/train_voc_5-3.sh

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='5-3'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for STAR-M

#NAME='ep60'

#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}


#NAME='phase_replay'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}