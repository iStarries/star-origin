import argparse
import random
import collections
import re
from pathlib import Path

import numpy as np
import socket
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

import models.model as module_arch
import utils.metric as module_metric
import utils.lr_scheduler as module_lr_scheduler
import data_loader.data_loaders as module_data
from trainer.trainer_voc import Trainer_base, Trainer_incremental
from utils.parse_config import ConfigParser
from logger.logger import Logger
from utils.memory import memory_sampling_balanced


def _pick_latest_by_epoch(candidates):
    epoch_re = re.compile(r"epoch(\d+)")
    best = None
    best_epoch = -1
    for path in candidates:
        match = epoch_re.search(path.name)
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best = path
    return best if best is not None else (candidates[-1] if candidates else None)


def _resolve_prev_checkpoint(save_dir: Path, prev_step: int) -> Path:
    """查找上一阶段的 checkpoint，优先选择最终 epoch 的权重。"""

    base_dir = save_dir.parent
    # 首选最后一个 epoch 的权重，其次才回退到历史最佳（兼容旧命名）
    candidates = sorted(base_dir.glob(f"step_{prev_step}_*/checkpoint-epoch*.pth"))

    if not candidates:
        candidates = sorted((base_dir / f"step_{prev_step}").glob(f"checkpoint-epoch*.pth"))

    chosen = _pick_latest_by_epoch(candidates)

    if chosen:
        return chosen

    candidates = sorted(base_dir.glob(f"step_{prev_step}_*/best-epoch*.pth"))

    if not candidates:
        candidates = sorted((base_dir / f"step_{prev_step}").glob("best-epoch*.pth"))

    chosen = _pick_latest_by_epoch(candidates)

    if chosen:
        return chosen

    raise FileNotFoundError(
        f"No checkpoint found for step {prev_step} under {base_dir}")

torch.backends.cudnn.benchmark = True


def main(config):
    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Single node, mutliple GPUs
        config.config['world_size'] = ngpus_per_node * config['world_size']
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Rather using distributed, use DataParallel
        main_worker(None, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config['multiprocessing_distributed']:
        config.config['rank'] = config['rank'] * ngpus_per_node + gpu

        # Resolve distributed URL to avoid port conflicts when the default port is busy
        if config['dist_url'] == 'auto':
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                free_port = s.getsockname()[1]
            config.config['dist_url'] = f"tcp://127.0.0.1:{free_port}"

        dist.init_process_group(
            backend=config['dist_backend'], init_method=config['dist_url'],
            world_size=config['world_size'], rank=config['rank']
        )
        rank = dist.get_rank()
    else:
        # 单卡不初始化分布式，保持 rank=0
        rank = 0
        config.config['rank'] = 0

    # Set looging
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f'train(rank{rank})', verbosity=2)

    run_info = config.config.get('info', '')
    if run_info:
        logger.info(f"Info: {run_info}")

    # fix random seeds for reproduce
    SEED = config['seed']
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Task information
    task_step = config['data_loader']['args']['task']['step']
    task_name = config['data_loader']['args']['task']['name']
    task_setting = config['data_loader']['args']['task']['setting']

    # Create Dataloader
    dataset = config.init_obj('data_loader', module_data)

    # Create old Model
    if task_step > 0:
        model_old = config.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes(task_step - 1)})
        if config['multiprocessing_distributed'] and (config['arch']['args']['norm_act'] == 'bn_sync'):
            model_old = nn.SyncBatchNorm.convert_sync_batchnorm(model_old)
    else:
        model_old = None

    # Memory pre-processing
    if (task_step > 0) and (config['data_loader']['args']['memory']['mem_size'] > 0):
        memory_sampling_balanced(
            config,
            model_old,
            dataset.get_old_train_loader(),
            ('voc', task_setting, task_name, task_step),
            logger, gpu,
        )
        dataset.get_memory(config, concat=True)
    logger.info(f"{str(dataset)}")
    logger.info(f"{dataset.dataset_info()}")

    if config['multiprocessing_distributed']:
        train_sampler = DistributedSampler(dataset.train_set)
    else:
        train_sampler = None

    train_loader = dataset.get_train_loader(train_sampler)
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()

    new_classes, old_classes = dataset.get_task_labels()
    logger.info(f"Old Classes: {old_classes}")
    logger.info(f"New Classes: {new_classes}")

    # Create Model
    model = config.init_obj('arch', module_arch, **{"classes": dataset.get_per_task_classes()})
    model._set_bn_momentum(model.backbone, momentum=0.01)

    # Convert BN to SyncBN for DDP
    if config['multiprocessing_distributed'] and (config['arch']['args']['norm_act'] == 'bn_sync'):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # logger.info(model)

    # Load previous step weights
    if task_step > 0:
        manual_prev = config.config.get('prev_best_checkpoint')
        if manual_prev is not None:
            old_path = Path(manual_prev)
            if not old_path.exists():
                raise FileNotFoundError(f"Provided prev_best_checkpoint does not exist: {old_path}")
            logger.info(f"Load weights from a user-provided previous step: {old_path}")
        else:
            old_path = _resolve_prev_checkpoint(config.save_dir, task_step - 1)
            config.config['prev_best_checkpoint'] = str(old_path)
            logger.info(f"Load weights from a previous step:{old_path}")

        model._load_pretrained_model(f'{old_path}')

        # Load old model to use KD
        if model_old is not None:
            model_old._load_pretrained_model(f'{old_path}')

        logger.info('** Random Initialization **')
    else:
        logger.info('Train from scratch')

    # Build optimizer
    if task_step > 0:
        optimizer = config.init_obj(
            'optimizer',
            torch.optim,
            [{"params": model.get_backbone_params()},
             {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10},
             {"params": model.get_old_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10},
             # {"params": model.get_backbone_params(), "weight_decay": 0},
             # {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
             # {"params": model.get_old_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10, "weight_decay": 0},
             {"params": model.get_new_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 100}]
        )
    else:
        optimizer = config.init_obj(
            'optimizer',
            torch.optim,
            [{"params": model.get_backbone_params()},
             {"params": model.get_aspp_params(), "lr": config["optimizer"]["args"]["lr"] * 10},
             {"params": model.get_classifer_params(), "lr": config["optimizer"]["args"]["lr"] * 10}]
        )

    lr_scheduler = config.init_obj(
        'lr_scheduler',
        module_lr_scheduler,
        **{"optimizer": optimizer, "max_iters": config["trainer"]['epochs'] * len(train_loader)}
    )

    evaluator_val = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes + 1, [0], new_classes]
    )

    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c

    evaluator_test = config.init_obj(
        'evaluator',
        module_metric,
        *[dataset.n_classes + 1, list(set(old_classes + [0])), new_classes]
    )

    if task_step > 0:
        trainer = Trainer_incremental(
            model=model, model_old=model_old,
            optimizer=optimizer,
            evaluator=(evaluator_val, evaluator_test),
            config=config,
            task_info=dataset.task_info(),
            data_loader=(train_loader, val_loader, test_loader),
            lr_scheduler=lr_scheduler,
            logger=logger, gpu=gpu,
        )
    else:
        trainer = Trainer_base(
            model=model,
            optimizer=optimizer,
            evaluator=(evaluator_val, evaluator_test),
            config=config,
            task_info=dataset.task_info(),
            data_loader=(train_loader, val_loader, test_loader),
            lr_scheduler=lr_scheduler,
            logger=logger, gpu=gpu,
        )

    logger.print(f"{torch.randint(0, 100, (1, 1))}")
    if dist.is_available() and dist.is_initialized():
        torch.distributed.barrier()

    trainer.train()
    trainer.test()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Class incremental Semantic Segmentation')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type action target', defaults=(None, float, None, None))
    options = [
        CustomArgs(['--multiprocessing_distributed'], action='store_true', target='multiprocessing_distributed'),
        CustomArgs(['--dist_url'], type=str, target='dist_url'),

        CustomArgs(['--name'], type=str, target='name'),
        CustomArgs(['--info'], type=str, target='info'),
        CustomArgs(['--save_dir'], type=str, target='trainer;save_dir'),

        CustomArgs(['--mem_size'], type=int, target='data_loader;args;memory;mem_size'),

        CustomArgs(['--seed'], type=int, target='seed'),
        CustomArgs(['--ep', '--epochs'], type=int, target='trainer;epochs'),
        CustomArgs(['--val_period', '--val_every'], type=int, target='trainer;validation_period'),
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;train;batch_size'),

        CustomArgs(['--task_name'], type=str, target='data_loader;args;task;name'),
        CustomArgs(['--task_step'], type=int, target='data_loader;args;task;step'),
        CustomArgs(['--task_setting'], type=str, target='data_loader;args;task;setting'),

        CustomArgs(['--pos_weight'], type=float, target='hyperparameter;pos_weight'),
        CustomArgs(['--mbce'], type=float, target='hyperparameter;mbce'),
        CustomArgs(['--mbce_distill'], type=float, target='hyperparameter;mbce_distill'),
        CustomArgs(['--kd'], type=float, target='hyperparameter;kd'),
        CustomArgs(['--ac'], type=float, target='hyperparameter;ac'),
        CustomArgs(['--enable_mbce_distill'], action='store_true', target='hyperparameter;enable_mbce_distill'),
        CustomArgs(['--distill_bg_only'], action='store_true', target='hyperparameter;distill_bg_only'),
        CustomArgs(['--use_consistency_filter'], action='store_true', target='hyperparameter;use_consistency_filter'),
        CustomArgs(['--consistency_old_thresh'], type=float, target='hyperparameter;consistency_old_thresh'),
        CustomArgs(['--consistency_curr_thresh'], type=float, target='hyperparameter;consistency_curr_thresh'),
        CustomArgs(['--use_separate_old_update'], action='store_true', target='hyperparameter;use_separate_old_update'),
        CustomArgs(['--pseudo_grad_scale'], type=float, target='hyperparameter;pseudo_grad_scale'),

        CustomArgs(['--freeze_bn'], action='store_true', target='arch;args;freeze_all_bn'),
        CustomArgs(['--test'], action='store_true', target='test'),

        CustomArgs(['--prev_best_checkpoint'], type=str, target='prev_best_checkpoint'),
        CustomArgs(['--prev_prototypes_path'], type=str, target='prev_prototypes_path'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
