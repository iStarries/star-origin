#!/bin/bash

# bash /media/wyh/star/voc.sh
#2x   bash /media/disk1/media/wyh/star/voc.sh

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


# --phase_replay
#${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS}
#${grad_ARGS}
#--validate --validate_a

####---------------------------------------------------------------------------------------------------------------
#
NAME='phase-gmm-lam0001-rs01-k1-warm10'
TASKNAME='5-3'
PHASE_ARGS="--phase_replay --phase_lambda 0.001 --phase_k_old 1 --phase_n_bins 8 \
--phase_residual_scale 0.1 --phase_sample_mode argmax --phase_gmm_components 3 \
--phase_gmm_min_var 1e-4 --phase_phase_noise 0.0 --phase_amp_noise 0.0 \
--phase_ref_mode batch_mean --phase_use_delta --phase_use_amp_stats --phase_warmup_epochs 10"
#
python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc.json --validate_a \
-d ${GPU} --save_dir ${SAVEDIR} --name ${NAME} ${PHASE_ARGS} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}


##---------------------------------------------------------------------------------------------------------------

#CUDA_VISIBLE_DEVICES=0 python eval_voc.py \
#  -c /media/disk1/media/wyh/star/checkpoints/ade_overlapped_50-50_STAR/config.json \
#  -r /media/disk1/media/wyh/star/checkpoints/ade_overlapped_50-50_STAR/overlapped_100-50_STAR.pth \
#  --device 0 --test
