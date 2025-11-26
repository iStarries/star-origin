import torch
import torch.nn as nn
import torch.nn.parallel
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import BCELoss, WBCELoss, PKDLoss, ContLoss
from data_loader import VOC

class Trainer_base(BaseTrainer):
    """
    Trainer class for a base step
    """
    def __init__(
        self, model, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None, visulized_dir=None
    ):
        super().__init__(config, logger, gpu)
        if not torch.cuda.is_available():
            logger.info("using CPU, this will be slow")
        elif config['multiprocessing_distributed']:
            if gpu is not None:
                torch.cuda.set_device(self.device)
                model.to(self.device)
                # When using a single GPU per process and per
                # DDP, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.model = DDP(model, device_ids=[gpu])  # Detectron: broadcast_buffers=False

            else:
                model.to(self.device)
                # DDP will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = DDP(model)

        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = nn.DataParallel(model, device_ids=self.device_ids)

        self.optimizer = optimizer
        self.evaluator_val = evaluator[0]
        self.evaluator_test = evaluator[1]

        self.task_info = task_info
        self.n_old_classes = len(self.task_info['old_class'])  # 0
        self.n_new_classes = len(self.task_info['new_class'])  # 19-1: 19 | 15-5: 15 | 15-1: 15...

        self.train_loader = data_loader[0]
        if self.train_loader is not None:
            self.len_epoch = len(self.train_loader)

        self.val_loader = data_loader[1]
        if self.val_loader is not None:
            self.do_validation = self.val_loader is not None

        self.test_loader = data_loader[2]
        if self.test_loader is not None:
            self.do_test = self.test_loader is not None

        self.lr_scheduler = lr_scheduler

        # For automatic mixed precision(AMP)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce',
            writer=self.writer,
            colums=['total', 'counts', 'average'],
        )
        self.valid_metrics = MetricTracker_scalars(writer=self.writer)
        self.test_metrics = MetricTracker_scalars(writer=self.writer)

        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        pos_weight = torch.ones(
            [len(self.task_info['new_class'])], device=self.device) * self.config['hyperparameter']['pos_weight']
        self.BCELoss = WBCELoss(
            pos_weight=pos_weight, n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.enable_mbce_distill = self.config['hyperparameter'].get('enable_mbce_distill', False)
        self.distill_bg_only = self.config['hyperparameter'].get('distill_bg_only', False)
        self.DistillBCELoss = BCELoss(reduction='none', distill_bg_only=self.distill_bg_only)

        self.mbce_weight = self.config['hyperparameter']['mbce']
        self.mbce_distill_weight = self.config['hyperparameter'].get('mbce_distill', self.mbce_weight)

        self._print_train_info()

        self.visulized_dir = visulized_dir

        if not config['test']:
            self.compute_cls_number(self.config)

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.mbce_weight} * L_mbce")
        if self.enable_mbce_distill:
            self.logger.info(
                f"          + {self.mbce_distill_weight} * L_mbce_distill"
                f" (bg_only={self.distill_bg_only})")
        else:
            self.logger.info("          + 0 * L_mbce_distill (disabled)")

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
        else:
            self.model.freeze_bn(affine_freeze=False)

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                logit, features, _ = self.model(data['image'], ret_intermediate=False)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                loss = self.mbce_weight * loss_mbce.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def _forward_loss_pass(self, data):
        """执行一次前向并计算增量学习各项损失。"""
        with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
            if self.model_old is not None:
                with torch.no_grad():
                    logit_old, features_old, _ = self.model_old(data['image'], ret_intermediate=True)

                    pred = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                    idx = (logit_old > 0.5).float()
                    idx = idx.sum(dim=1)
                    pred[idx == 0] = 0
                    pseudo_label_region_base = torch.logical_and(
                        data['label'] == 0, pred > 0
                    ).unsqueeze(1)
            else:
                logit_old, features_old = None, None
                pseudo_label_region_base = torch.zeros_like(data['label'], dtype=torch.bool).unsqueeze(1)

            fake_features = []
            for cls in range(0, self.per_iter_prev_number.shape[0]):
                per_cls_fake_features = \
                    self.prev_prototypes[cls].reshape(1, -1, 1, 1).repeat(1, 1, self.per_iter_prev_number[cls], 1)
                noise = \
                    torch.randn_like(per_cls_fake_features) * self.prev_noise[cls].reshape(1, -1, 1, 1)
                per_cls_fake_features = per_cls_fake_features + noise
                rand_norm = \
                    torch.randn_like(per_cls_fake_features) * \
                    self.prev_norm[1, cls].reshape(1, 1, 1, 1) + \
                    self.prev_norm[0, cls].reshape(1, 1, 1, 1)
                per_cls_fake_features = per_cls_fake_features * rand_norm
                fake_features.append(per_cls_fake_features)

            fake_features = torch.cat(fake_features, dim=2)
            fake_label = torch.zeros(1, fake_features.shape[2], 1, requires_grad=False).to(self.device)

            if self.model_old is not None:
                region_bg = torch.logical_and(pred == 0, data['label'] == 0)[:, 8::16, 8::16]
            else:
                region_bg = torch.zeros_like(data['label'][:, 8::16, 8::16], dtype=torch.bool)

            logit, features, extra = \
                self.model(data['image'], ret_intermediate=True, fake_features=fake_features, region_bg=region_bg)
            logits_for_fake = extra[0]
            logits_for_extra_bg = extra[1]

            # 一致性过滤：旧模型与当前模型的预测一致且置信度均高时，才保留伪标签区域。
            if self.use_consistency_filter and (logit_old is not None):
                with torch.no_grad():
                    old_prob = torch.sigmoid(logit_old)
                    old_conf, old_pred = old_prob.max(dim=1)

                current_prob = torch.sigmoid(logit.detach())
                curr_conf, curr_pred = current_prob.max(dim=1)
                consistency_mask = (old_pred == curr_pred) & \
                    (old_conf > self.consistency_old_thresh) & \
                    (curr_conf > self.consistency_curr_thresh)

                pseudo_label_region = pseudo_label_region_base & consistency_mask.unsqueeze(1)
                kept = pseudo_label_region.sum()
                total = pseudo_label_region_base.sum() + 1e-6
                consistency_ratio = (kept / total).detach()
            else:
                pseudo_label_region = pseudo_label_region_base
                consistency_ratio = torch.tensor(1.0, device=self.device)

            # [|Ct|]
            loss_mbce_ori = self.BCELoss(
                logit[:, -self.n_new_classes:],
                data['label'],
            ).mean(dim=[0, 2, 3])

            if logits_for_extra_bg is not None:
                extra_bg_label = torch.zeros(1, logits_for_extra_bg.shape[2], 1, requires_grad=False).to(self.device)
                loss_mbce_extra_bg = self.BCELoss_extra_bg(
                    logits_for_extra_bg[:, -self.n_new_classes:],
                    extra_bg_label,
                ).mean(dim=[0, 2, 3])
            else:
                loss_mbce_extra_bg = torch.zeros_like(loss_mbce_ori)

            loss_mbce_fake = self.BCELoss_fake(
                logits_for_fake[:, -self.n_new_classes:],
                fake_label
            ).mean(dim=[0, 2, 3])

            stride_num = features[-1].shape[0] * features[-1].shape[2] * features[-1].shape[3]
            weight_extra_bg = self.extra_bg_ratio * region_bg.sum() / stride_num
            weight_fake = fake_label.shape[1] / stride_num

            loss_mbce = loss_mbce_ori + loss_mbce_fake * weight_fake + loss_mbce_extra_bg * weight_extra_bg
            loss_mbce = loss_mbce / (1 + weight_extra_bg + weight_fake)

            if self.enable_mbce_distill and logit_old is not None:
                loss_mbce_distill = self.DistillBCELoss(
                    logit, data['label'], logit_old).mean(dim=[0, 2, 3])
            else:
                loss_mbce_distill = torch.zeros_like(loss_mbce)

            if features_old is not None and pseudo_label_region.sum() > 0:
                loss_pkd = self.PKDLoss(features, features_old, pseudo_label_region.to(torch.float32))
            else:
                loss_pkd = torch.tensor(0.0, device=self.device)

            loss_cont = self.ContLoss(
                features[-1], logit[:, -self.n_new_classes:], data['label'], self.prev_prototypes)

        return {
            'loss_mbce': loss_mbce,
            'loss_mbce_distill': loss_mbce_distill,
            'loss_pkd': loss_pkd,
            'loss_cont': loss_cont,
            'consistency_ratio': consistency_ratio,
        }

    def _valid_epoch(self, epoch):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        log = {}
        self.evaluator_val.reset()
        self.logger.info(f"Number of val loader: {len(self.val_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, _, _ = self.model(data['image'])

                logit = torch.sigmoid(logit)
                pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_val.add_batch(target, pred)

            if self.rank == 0:
                self.writer.set_step((epoch), 'valid')

            for met in self.metric_ftns_val:
                if len(met().keys()) > 2:
                    self.valid_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                else:
                    self.valid_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': met()['old']})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': met()['new']})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': met()['harmonic']})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': met()['overall']})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_val.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        elif i in self.evaluator_val.old_classes_idx:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log

    def _test(self, epoch=None):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        log = {}
        self.evaluator_test.reset()
        self.logger.info(f"Number of test loader: {len(self.test_loader)}")

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                target = data['label'].cpu().numpy()

                logit, features, _ = self.model(data['image'])
                logit = torch.sigmoid(logit)
                pred = logit.argmax(dim=1) + 1  # pred: [N. H, W]

                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]

                pred[idx == 0] = 0  # set background (non-target class)

                pred = pred.cpu().numpy()
                self.evaluator_test.add_batch(target, pred)

                self.progress(self.logger, batch_idx, len(self.test_loader))

            if epoch is not None:
                if self.rank == 0:
                    self.writer.set_step((epoch), 'test')

            for met in self.metric_ftns_test:
                if epoch is not None:
                    if len(met().keys()) > 2:
                        self.test_metrics.update(met.__name__, [met()['old'], met()['new'], met()['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met()['overall']], 'overall', n=1)

                if 'old' in met().keys():
                    log.update({met.__name__ + '_old': f"{met()['old']:.2f}"})
                if 'new' in met().keys():
                    log.update({met.__name__ + '_new': met()['new']})
                if 'harmonic' in met().keys():
                    log.update({met.__name__ + '_harmonic': met()['harmonic']})
                if 'overall' in met().keys():
                    log.update({met.__name__ + '_overall': f"{met()['overall']:.2f}"})
                if 'by_class' in met().keys():
                    by_class_str = '\n'
                    for i in range(len(met()['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met()['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met()['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        return log


class Trainer_incremental(Trainer_base):
    """
    Trainer class for incremental steps
    """
    def __init__(
        self, model, model_old, optimizer, evaluator, config, task_info,
        data_loader, lr_scheduler=None, logger=None, gpu=None
    ):
        super().__init__(
            model=model, optimizer=optimizer, evaluator=evaluator, config=config, task_info=task_info,
            data_loader=data_loader, lr_scheduler=lr_scheduler, logger=logger, gpu=gpu)

        if config['multiprocessing_distributed']:
            if gpu is not None:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old, device_ids=[gpu])
            else:
                if model_old is not None:
                    model_old.to(self.device)
                    self.model_old = DDP(model_old)
        else:
            if model_old is not None:
                self.model_old = nn.DataParallel(model_old, device_ids=self.device_ids)

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_pkd', 'loss_cont', 'loss_bce_distill',
            'loss_old_step', 'consistency_ratio',
            writer=self.writer, colums=['total', 'counts', 'average'],
        )
        if config.resume is not None:
            self._resume_checkpoint(config.resume, config['test'])

        self.BCELoss_fake = WBCELoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.BCELoss_extra_bg = WBCELoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)
        self.DistillBCELoss = BCELoss(reduction='none')
        self.PKDLoss = PKDLoss()
        self.ContLoss = ContLoss(n_old_classes=self.n_old_classes + 1, n_new_classes=self.n_new_classes)

        # 旧类伪梯度/蒸馏相关配置
        hp = self.config['hyperparameter']
        self.use_consistency_filter = hp.get('use_consistency_filter', False)
        self.consistency_old_thresh = hp.get('consistency_old_thresh', 0.0)
        self.consistency_curr_thresh = hp.get('consistency_curr_thresh', 0.0)
        self.use_separate_old_update = hp.get('use_separate_old_update', False)
        self.pseudo_grad_scale = hp.get('pseudo_grad_scale', 1.0)

        prev_info_path = self._resolve_prev_info_path(config)

        prev_info = torch.load(prev_info_path)
        self.prev_numbers = prev_info['numbers'].to(self.device)
        self.prev_prototypes = prev_info['prototypes'].to(self.device)
        self.prev_norm = prev_info['norm_mean_and_std'].to(self.device)
        self.prev_noise = prev_info['noise'].to(self.device)

        self.current_numbers = self.numbers[1:].sum()

        assert task_info['setting'] in ['overlap', 'disjoint']

        if task_info['setting'] == 'overlap':
            self.prev_bg_number = self.prev_numbers[0] * (1 - 0.01 * self.n_new_classes)
        else:
            self.prev_bg_number = self.prev_numbers[0]

    def _print_train_info(self):
        self.logger.info(f"pos_weight - {self.config['hyperparameter']['pos_weight']}")
        self.logger.info(f"Total loss = {self.config['hyperparameter']['mbce']} * L_mbce + "
                         f"{self.config['hyperparameter']['pkd']} * L_pkd")

    def _resolve_prev_info_path(self, config):
        prev_best_checkpoint = config.config.get('prev_best_checkpoint', None)

        if prev_best_checkpoint is None:
            prev_step = config['data_loader']['args']['task']['step'] - 1
            prev_dir_candidates = sorted(Path(config.save_dir).parent.glob(f"step_{prev_step}_*"))

            if prev_dir_candidates:
                prev_dir = prev_dir_candidates[-1]
            else:
                prev_dir = Path(config.save_dir).parent / f"step_{prev_step}"

            target_epoch = config['trainer']['epochs']
        else:
            prev_dir = Path(prev_best_checkpoint).parent
            prev_checkpoint = torch.load(prev_best_checkpoint, map_location='cpu')
            target_epoch = prev_checkpoint.get('epoch', config['trainer']['epochs'])

        return prev_dir / f"prototypes-epoch{target_epoch}.pth"

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        self.model.train()
        if isinstance(self.model, (nn.DataParallel, DDP)):
            self.model.module.freeze_bn(affine_freeze=False)
            self.model.module.freeze_dropout()
        else:
            self.model.freeze_bn(affine_freeze=False)
            self.model.freeze_dropout()
        self.model_old.eval()

        self.train_metrics.reset()
        self.logger.info(f'Epoch - {epoch}')

        # Random shuffling
        if not isinstance(self.train_loader.sampler, torch.utils.data.RandomSampler):
            self.train_loader.sampler.set_epoch(epoch)

        if epoch == 1:
            self.pred_numbers = self.compute_pred_number()
            if self.task_info['setting'] == 'overlap':
                self.per_iter_prev_number = \
                    ((self.prev_numbers[1:] - self.pred_numbers[1:]) / len(self.train_loader)).to(torch.int)
            else:
                self.per_iter_prev_number = \
                    (self.prev_numbers[1:] / len(self.train_loader)).to(torch.int)

            self.per_iter_prev_number = torch.clamp(self.per_iter_prev_number, min=0)

            if self.task_info['setting'] == 'overlap':
                self.extra_bg_ratio = (self.prev_bg_number / self.pred_numbers[0]) - 0.5
            else:
                self.extra_bg_ratio = (self.prev_bg_number / self.pred_numbers[0])

            self.extra_bg_ratio = torch.clamp(self.extra_bg_ratio, min=0)

            tot_numbers = self.prev_numbers.clone().to(self.device)
            if self.task_info['setting'] == 'overlap':
                tot_numbers[0] = self.pred_numbers[0] * 0.5 + self.prev_bg_number
            else:
                tot_numbers[0] = self.pred_numbers[0] + self.prev_bg_number
                tot_numbers[1:] = self.prev_numbers[1:].to(self.device) + self.pred_numbers[1:]

            tot_numbers = torch.cat((tot_numbers.to(self.device), self.numbers[1:].to(self.device)), dim=0)
            self.numbers = tot_numbers
        
        for batch_idx, data in enumerate(self.train_loader):
            data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            # 第一步：主监督损失
            loss_dict = self._forward_loss_pass(data)

            distill_weight = self.mbce_distill_weight if self.enable_mbce_distill else 0
            loss_main = self.mbce_weight * loss_dict['loss_mbce'].sum()
            loss_old = distill_weight * loss_dict['loss_mbce_distill'].sum() \
                       + self.config['hyperparameter']['pkd'] * loss_dict['loss_pkd'].sum() \
                       + self.config['hyperparameter']['cont'] * loss_dict['loss_cont']

            if not self.use_separate_old_update:
                loss = loss_main + loss_old
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 主监督更新
                self.scaler.scale(loss_main).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # 独立的旧类伪梯度/蒸馏更新
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict_old = self._forward_loss_pass(data)
                loss_old_step = distill_weight * loss_dict_old['loss_mbce_distill'].sum() \
                                 + self.config['hyperparameter']['pkd'] * loss_dict_old['loss_pkd'].sum() \
                                 + self.config['hyperparameter']['cont'] * loss_dict_old['loss_cont']

                if loss_old_step.requires_grad:
                    loss_old_scaled = self.pseudo_grad_scale * loss_old_step
                    self.scaler.scale(loss_old_scaled).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                loss_old = loss_old_step

            # 统一的 lr_scheduler 步进
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', (loss_main + loss_old).item())
            self.train_metrics.update('loss_mbce', loss_dict['loss_mbce'].sum().item() * self.mbce_weight)
            self.train_metrics.update('loss_bce_distill',
                                      loss_dict['loss_mbce_distill'].sum().item() * distill_weight)
            self.train_metrics.update('loss_pkd', loss_dict['loss_pkd'].sum().item() * self.config['hyperparameter']['pkd'])
            self.train_metrics.update('loss_cont', loss_dict['loss_cont'].item() * self.config['hyperparameter']['cont'])
            self.train_metrics.update('loss_old_step', loss_old.item() if torch.is_tensor(loss_old) else float(loss_old))
            self.train_metrics.update('consistency_ratio', loss_dict['consistency_ratio'].item())

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            self.progress(self.logger, batch_idx, len(self.train_loader))

            if batch_idx == self.len_epoch:
                break

        # average train loss per epoch
        log = self.train_metrics.result()

        val_flag = False
        if self.do_validation and (epoch % self.validation_period) == 0:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.rank == 0:
                val_flag = True

        return log, val_flag

    def compute_pred_number(self):
        self.logger.info("computing pred number of pixels...")

        pred_numbers = torch.zeros(self.n_old_classes + 1).to(self.device)

        for batch_idx, data in enumerate(self.train_loader):
            with torch.no_grad():
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                logit_old, _, _ = self.model_old(data['image'], ret_intermediate=False)

                logit_old = logit_old.detach()
                pred = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                idx = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0  # set background (non-target class)
                pred_region = (pred * (data['label'] == 0))[:, 8::16, 8::16]

                real_bg_region = torch.logical_and(pred == 0, data['label'] == 0)[:, 8::16, 8::16]
                pred_numbers[0] = pred_numbers[0] + real_bg_region.sum()

                for i in range(1, self.n_old_classes + 1):
                    pred_numbers[i] = pred_numbers[i] + (pred_region == i).sum()

            self.progress(self.logger, batch_idx, len(self.train_loader))

        return pred_numbers
