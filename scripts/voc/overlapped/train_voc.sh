#!/bin/bash

# bash /media/wyh/star/scripts/voc/overlapped/train_voc.sh
#2x   bash /media/disk1/media/wyh/star/scripts/voc/overlapped/train_voc.sh

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='overlap'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for STAR-M
CONSISTENCY_ARGS="--use_consistency_filter --consistency_old_thresh 0.7 --consistency_curr_thresh 0.6"
SEPARATE_UPDATE_ARGS="--use_separate_old_update --pseudo_grad_scale 1.0"
grad_ARGS="--grad --grad_samples 512 --grad_hidden 64 --grad_layers 2 --grad_alpha 0.5 --grad_eta 1.0 --grad_lambda 1.0 --grad_eps 1e-6 --grad_lr 1e-3 --grad_warmup 0"

#CUDA_VISIBLE_DEVICES=0 python eval_voc.py \
#  -c /media/wyh/star/saved_voc/models/overlap_15-5_real-grad-test/step_1_20251205-132419/config.json \
#  -r /media/wyh/star/saved_voc/models/overlap_15-5_real-grad-test/step_1_20251205-132419/best-test-epoch47-miou74.52.pth \
#  --device 0 --test



#--epoch 20
#--phase_replay
#${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}
#${grad_ARGS}
#--validate

#NAME='real-grad-test'
#TASKNAME='15-5'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} ${grad_ARGS}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} ${grad_ARGS} --validate --freeze_bn --mem_size ${MEMORY_SIZE}
#
#NAME='phase-test'
#TASKNAME='15-5'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay --validate --freeze_bn --mem_size ${MEMORY_SIZE}
#
#NAME='grad-test'
#TASKNAME='15-5'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --validate --freeze_bn --mem_size ${MEMORY_SIZE}
#
####---------------------------------------------------------------------------------------------------------------
#
#
NAME='grad-test'
TASKNAME='5-3'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --validate --freeze_bn --mem_size ${MEMORY_SIZE}


##---------------------------------------------------------------------------------------------------------------

#NAME='ep60-Phase-test'
#TASKNAME='5-3'
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
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --validate --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
