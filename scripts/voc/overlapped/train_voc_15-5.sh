#!/bin/bash
# bash /media/wyh/star/scripts/voc/overlapped/train_voc_15-5.sh
# bash /media/disk1/media/wyh/star/scripts/voc/overlapped/train_voc_15-5.sh
# GPU=0
# BS=24
# SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='15-5'
INIT_LR=0.001
LR=0.0001
# 如需开启 BCE 蒸馏，直接在命令末尾追加： --enable_mbce_distill [--distill_bg_only]
# --mem_size 50
# --epochs 60
NAME='STAR'
# --mbce_distill 0.5
# --prev_best_checkpoint /path/to/step0/best-epoch60.pth
# --prev_prototypes_path /path/to/step0/prototypes-epoch60.pth
#------------------------------------------------------------------------------------------------------------------original

#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${INIT_LR} \
#--task_name '15-5' --task_step 0 \
#--bs 24 --epochs 80 --val_every 2

#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 80 --val_every 2

#CUDA_VISIBLE_DEVICES=0 python eval_voc.py \
#  -c /media/wyh/star/saved_voc/models/overlap_15-5_epochs80/step_1_20251127-142820/config.json \
#  -r /media/wyh/star/saved_voc/models/overlap_15-5_epochs80/step_1_20251127-142820/best-epoch67-val_Mean_Intersection_over_Union_overall-76.9194.pth \
#  --device 0 --test
#------------------------------------------------------------------------------------------------------------------original


#-----------------------------------------------梯度学习器
# 一致性过滤与分离式更新的可调开关
#CONSISTENCY_ARGS="--use_consistency_filter --consistency_old_thresh 0.5 --consistency_curr_thresh 0.5"
#SEPARATE_UPDATE_ARGS="--use_separate_old_update --pseudo_grad_scale 1.0"
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${INIT_LR} \
#--task_name '15-1' --task_step 0 --name 'continue-gradient05' \
#--bs 24 --epochs 60 --val_every 10
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn --mem_size 0 \
#--task_name '15-1' --task_step 1 --name 'continue-gradient05' ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
#--bs 24 --epochs 60 --val_every 10
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn --mem_size 0 \
#--task_name '15-1' --task_step 2 --name 'continue-gradient05' ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
#--bs 24 --epochs 60 --val_every 10
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn --mem_size 0 \
#--task_name '15-1' --task_step 3 --name 'continue-gradient05' ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
#--bs 24 --epochs 60 --val_every 10
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn --mem_size 0 \
#--task_name '15-1' --task_step 4 --name 'continue-gradient05' ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
#--bs 24 --epochs 60 --val_every 10
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn --mem_size 0 \
#--task_name '15-1' --task_step 5 --name 'continue-gradient05' ${CONSISTENCY_ARGS} ${SEPARATE_UPDATE_ARGS} \
#--bs 24 --epochs 60 --val_every 10

#-----------------------------------------------梯度学习器


# 给仓库里上传了上一次训练的两个步骤的日志文件，分析一个问题：为什么第二个步骤学习新类后测试都性能如此低，而第一个步骤学习的旧类效果显著。


#------------------------------------------------------软标签
####原生代码60次迭代效果有没有保持原样
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name 'epochs80' --task_setting ${TASKSETTING} --lr ${INIT_LR} \
#--task_name '15-5' --task_step 0 \
#--bs 24 --epochs 80 --val_every 1 --info 'train0-bs24-epochs80'
#
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name 'epochs80' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 80 \
#--val_every 1 --info 'train1-bs24-epochs80'
##
####原生60次迭代加上软标签和过滤背景软标签有没有变好
#python train_voc.py -c configs/config_voc.json \
#-d 0 --save_dir 'saved_voc' --name ${NAME} --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
#--task_name '15-5' --task_step 1 \
#--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill \
#--val_every 2 --info 'train1-enable_mbce_distill'
##
python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name 'continue-distill-weighted_1' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-1' --task_step 1 \
--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --mbce_distill 1 \
--val_every 10 \
--prev_best_checkpoint '/media/disk1/media/wyh/star/saved_voc/models/overlap_15-5_STAR/step_0_ep60/checkpoint-epoch60-val_Mean_Intersection_over_Union_overall-80.3113.pth' \
--prev_prototypes_path '/media/disk1/media/wyh/star/saved_voc/models/overlap_15-5_STAR/step_0_ep60/prototypes-epoch60.pth'

python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name 'continue-distill-weighted_1' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-1' --task_step 2 \
--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --mbce_distill 1 \
--val_every 10 \

python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name 'continue-distill-weighted_1' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-1' --task_step 3 \
--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --mbce_distill 1 \
--val_every 10 \

python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name 'continue-distill-weighted_1' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-1' --task_step 4 \
--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --mbce_distill 1 \
--val_every 10 \

python train_voc.py -c configs/config_voc.json \
-d 0 --save_dir 'saved_voc' --name 'continue-distill-weighted_1' --task_setting ${TASKSETTING} --lr ${LR} --freeze_bn \
--task_name '15-1' --task_step 5 \
--bs 24 --mem_size 0 --epochs 60 --enable_mbce_distill --mbce_distill 1 \
--val_every 10 \

#------------------------------------------------------软标签







