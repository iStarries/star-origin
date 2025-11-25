#!/bin/bash

GPU=0
BS=24
SAVEDIR='saved_voc'

TASKSETTING='disjoint'
TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
# Per-step memory sizes can be overridden before invoking this script, e.g.
# MEM_SIZE_STEP0=0 MEM_SIZE_STEP1=50 ./train_voc_15-5.sh
MEM_SIZE_STEP0=${MEM_SIZE_STEP0:-0}
MEM_SIZE_STEP1=${MEM_SIZE_STEP1:-0} # 50 for STAR-M

# Optional distillation flags per step, e.g.
# DISTILL_ARGS_STEP1="--enable_mbce_distill --distill_bg_only"
DISTILL_ARGS_STEP0=${DISTILL_ARGS_STEP0:-""}
DISTILL_ARGS_STEP1=${DISTILL_ARGS_STEP1:-""}

NAME='STAR'
python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} \
--mem_size ${MEM_SIZE_STEP0} ${DISTILL_ARGS_STEP0}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn \
--mem_size ${MEM_SIZE_STEP1} ${DISTILL_ARGS_STEP1}



