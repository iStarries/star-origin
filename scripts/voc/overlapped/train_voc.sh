#!/bin/bash

# bash /media/wyh/star/scripts/voc/overlapped/train_voc_15-1.sh
#2x   bash /media/disk1/media/wyh/star/scripts/voc/overlapped/train_voc_15-1.sh

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='overlap'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for STAR-M
CONSISTENCY_ARGS="--use_consistency_filter --consistency_old_thresh 0.7 --consistency_curr_thresh 0.6"
SEPARATE_UPDATE_ARGS="--use_separate_old_update --pseudo_grad_scale 1.0"

#--epoch 60
#--phase_replay
#${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}

NAME='ep20'
TASKNAME='15-1'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --epoch 20

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --epoch 20 --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --epoch 20 --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --epoch 20 --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --epoch 20 --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --epoch 20 --freeze_bn --mem_size ${MEMORY_SIZE}

#---------------------------------------------------------------------------------------------------------------

NAME='ep20-grad'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --epoch 20

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --epoch 20 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --epoch 20 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --epoch 20 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --epoch 20 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --epoch 20 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

