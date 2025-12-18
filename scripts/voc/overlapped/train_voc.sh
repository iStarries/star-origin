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

####---------------------------------------------------------------------------------------------------------------
#
#
#NAME='phase-update-replay'
#TASKNAME='15-1'
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
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} --validate \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}

NAME='phase-append'
TASKNAME='15-1'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --phase_replay --validate --freeze_bn --mem_size ${MEMORY_SIZE}









##---------------------------------------------------------------------------------------------------------------

#CUDA_VISIBLE_DEVICES=0 python eval_voc.py \
#  -c /media/disk1/media/wyh/star/checkpoints/ade_overlapped_50-50_STAR/config.json \
#  -r /media/disk1/media/wyh/star/checkpoints/ade_overlapped_50-50_STAR/overlapped_100-50_STAR.pth \
#  --device 0 --test

#CUDA_VISIBLE_DEVICES=0 python eval_boundary_voc.py \
#  -c /media/wyh/star/duibi/overlap_15-1_real-grad-spatial/step_5_20251204-142715/config.json \
#  -r /media/wyh/star/duibi/overlap_15-1_real-grad-spatial/step_5_20251204-142715/best-test-epoch49-miou72.95.pth \
#  --boundary_width 3 --min_mask_pixels 50

#python scripts/plot_boundary_curves.py \
#  --data-json results/boundary_voc15_1_old_classes.json \
#  --output boundary_old15_steps.png
