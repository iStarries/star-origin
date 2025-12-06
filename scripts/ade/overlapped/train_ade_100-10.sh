#!/bin/bash
#bash /media/wyh/star/scripts/ade/overlapped/train_ade_100-10.sh
#2x   bash /media/disk1/media/wyh/star/scripts/ade/overlapped/train_ade_100-10.sh
GPU=0
BS=10  # Total 24
SAVEDIR='saved_ade'

TASKSETTING='overlap'
TASKNAME='100-10'
INIT_LR=0.0021
LR=0.00021
MEMORY_SIZE=0

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

##21
#NAME='ep100-test'
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} \
#--config /media/disk1/media/wyh/star/saved_ade/models/overlap_100-10_ep100-test/step_0_20251210-141203/config.json \
#--resume /media/disk1/media/wyh/star/saved_ade/models/overlap_100-10_ep100-test/step_0_20251210-141203/checkpoint-epoch80.pth
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} --validate \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
##
#NAME='phase-test'
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --phase_replay
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} --validate \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --phase_replay --freeze_bn --mem_size ${MEMORY_SIZE}
#
#23
NAME='grad-test'
python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
--config /media/wyh/star/saved_ade/models/overlap_100-10_grad-test/step_0_20251210-192750/config.json \
--resume /media/wyh/star/saved_ade/models/overlap_100-10_grad-test/step_0_20251210-192750/checkpoint-epoch80.pth

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_ade.py -c configs/config_ade.json \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} --validate \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}


#24
#NAME='real-grad-test'
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} ${grad_ARGS} \
#--config /media/wyh/star/saved_ade/models/overlap_100-10_real-grad-test/step_0_20251210-180911/config.json \
#--resume /media/wyh/star/saved_ade/models/overlap_100-10_real-grad-test/step_0_20251210-180911/checkpoint-epoch70.pth
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_ade.py -c configs/config_ade.json \
#-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} --validate \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} ${grad_ARGS} --freeze_bn --mem_size ${MEMORY_SIZE}
