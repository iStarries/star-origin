#!/bin/bash

# bash /media/wyh/overlap_15-1_ep60/scripts/voc/overlapped/train_voc.sh
#2x   bash /media/disk1/media/wyh/overlap_15-1_ep60/scripts/voc/overlapped/train_voc.sh

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
thresh="--use_consistency_filter --consistency_thresh 0.7"

#CUDA_VISIBLE_DEVICES=0 python eval_voc.py \
#  -c /media/wyh/overlap_15-1_ep60/saved_voc/models/overlap_15-5_real-grad-test/step_1_20251205-132419/config.json \
#  -r /media/wyh/overlap_15-1_ep60/saved_voc/models/overlap_15-5_real-grad-test/step_1_20251205-132419/best-test-epoch47-miou74.52.pth \
#  --device 0 --test

#CUDA_VISIBLE_DEVICES=0 python eval_boundary_voc.py \
#  -c /path/to/step1/config.json \
#  -r /path/to/step1/checkpoint-epoch60.pth \
#  --boundary_width 3 --min_mask_pixels 50


#--epoch 20
#--phase_replay
#旧版一致性过滤：   ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}
#梯度学习器：   ${grad_ARGS}
#启用逐类阈值，缺失类回退到全局阈值： ${thresh}
#尾端验证：   --validate

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

####---------------------------------------------------------------------------------------------------------------
#
#
NAME='Phase+realgrad2'
TASKNAME='15-1'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay ${grad_ARGS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --validate --freeze_bn --mem_size ${MEMORY_SIZE}

##---------------------------------------------------------------------------------------------------------------

NAME='Phase+realgrad'
TASKNAME='5-3'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay ${grad_ARGS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --validate --phase_replay ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
NAME='Phase+realgrad+filter'
TASKNAME='5-3'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --validate --phase_replay ${grad_ARGS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#