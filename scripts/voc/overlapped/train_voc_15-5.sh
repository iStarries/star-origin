#!/bin/bash
# bash /media/wyh/star/scripts/voc/overlapped/train_voc_15-5.sh
# GPU=0
# BS=24
# SAVEDIR='saved_voc'

TASKSETTING='overlap'
# TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
# MEMORY_SIZE=0 # 50 for STAR-M
# 如需开启 BCE 蒸馏，直接在命令末尾追加： --enable_mbce_distill [--distill_bg_only]
# --mem_size 50
# --epochs 80

# 一致性过滤与分离式更新的可调开关
CONSISTENCY_ARGS="--use_consistency_filter --consistency_old_thresh 0.7 --consistency_curr_thresh 0.7"

# 统一的梯度学习器开关；设为0即可彻底关闭（不传递相关参数）
USE_GRADIENT_LEARNER=1
PSEUDO_GRAD_SCALE=1.0

if [ ${USE_GRADIENT_LEARNER} -eq 1 ]; then
    SEPARATE_UPDATE_ARGS="--use_separate_old_update --pseudo_grad_scale ${PSEUDO_GRAD_SCALE}"
else
    SEPARATE_UPDATE_ARGS=""
fi

NAME='STAR'
python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${INIT_LR} \
--task_name '15-5' --task_step 0 \
--bs 8 --epochs 80

python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-5' --task_step 1 ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
--bs 8 --mem_size 0 --epochs 80
