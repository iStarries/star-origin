import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP
from base import BaseTrainer
from utils import MetricTracker, MetricTracker_scalars
from models.loss import BCELoss, WBCELoss, PKDLoss, ContLoss
from models.gradient_learner import GradientLearner
from data_loader import VOC
from utils.prototype_bank import PhasePrototypeBank
from utils.fft_utils import (
    decompose_spectrum,
    extract_mid,
    get_mid_slices,
    reconstruct_feature,
    replace_mid,
)

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

        self.grad_cfg = self.config['hyperparameter'].get('grad_learner', {})
        self.grad_enabled = self.grad_cfg.get('enabled', False)
        self.grad_warmup = self.grad_cfg.get('warmup_epochs', 0)
        self.grad_lambda = self.grad_cfg.get('lambda_fit', 1.0)
        self.grad_alpha = self.grad_cfg.get('alpha', 0.5)
        self.grad_eta = self.grad_cfg.get('eta', 1.0)
        self.grad_epsilon = self.grad_cfg.get('epsilon', 1e-6)
        self.grad_sample_pixels = self.grad_cfg.get('sample_pixels', 256)

        if self.grad_enabled:
            grad_input_dim = self.model.module.tot_classes if isinstance(self.model, (nn.DataParallel, DDP)) else self.model.tot_classes
            grad_hidden = self.grad_cfg.get('hidden_dim', 64)
            grad_layers = self.grad_cfg.get('num_layers', 2)

            grad_net = GradientLearner(grad_input_dim, hidden_dim=grad_hidden, num_layers=grad_layers)
            grad_net = grad_net.to(self.device)

            if self.config['multiprocessing_distributed']:
                if gpu is not None:
                    self.grad_learner = DDP(grad_net, device_ids=[gpu])
                else:
                    self.grad_learner = DDP(grad_net)
            else:
                self.grad_learner = nn.DataParallel(grad_net, device_ids=self.device_ids)

            grad_lr = self.grad_cfg.get('lr', self.config['optimizer']['args']['lr'])
            self.grad_optimizer = torch.optim.Adam(self.grad_learner.parameters(), lr=grad_lr)
        else:
            self.grad_learner = None
            self.grad_optimizer = None

        if self.evaluator_val is not None:
            self.metric_ftns_val = [getattr(self.evaluator_val, met) for met in config['metrics']]
        if self.evaluator_test is not None:
            self.metric_ftns_test = [getattr(self.evaluator_test, met) for met in config['metrics']]

        self.train_metrics = MetricTracker(
            'loss', 'loss_mbce', 'loss_grad',
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

        # Phase replay configuration (disabled by default)
        phase_cfg = self.config['hyperparameter'].get('phase_replay', {})
        self.phase_replay_enabled = phase_cfg.get('enabled', False)
        self.phase_ratio = phase_cfg.get('phase_ratio', 1.0)
        self.phase_loss_weight = phase_cfg.get('phase_loss_weight', 1.0)

        # Feature-map replay configuration (Direction C)
        map_cfg = phase_cfg.get('map_replay', {})
        self.map_replay_enabled = map_cfg.get('enabled', False)
        self.map_replay_num_maps = int(map_cfg.get('num_maps_per_iter', 0))
        self.map_replay_weight = float(map_cfg.get('weight', 1.0))
        self.map_replay_w_distill = float(map_cfg.get('w_distill', 1.0))
        self.map_replay_w_new0 = float(map_cfg.get('w_new0', 1.0))
        self.map_replay_w_grad = float(map_cfg.get('w_grad', 0.0))
        self.map_replay_w_local = float(map_cfg.get('w_local', 0.0))
        self.map_replay_teacher_conf = float(map_cfg.get('teacher_conf_thresh', 0.0))
        self.map_replay_eps = float(map_cfg.get('eps', 1e-6))
        self.map_replay_use_pred_first = bool(map_cfg.get('use_pred_first', True))
        proto_sampling_cfg = phase_cfg.get('proto_sampling', {})
        self.phase_proto_sampling_min_norm = proto_sampling_cfg.get('min_norm_eps', 1e-6)
        self.phase_proto_sampling_max_trials = proto_sampling_cfg.get('max_trials', 3)
        self.phase_proto_amp_scale = proto_sampling_cfg.get('amp_scale', 1.0)
        if self.phase_replay_enabled:
            self.phase_bank = PhasePrototypeBank(
                mid_ratio=phase_cfg.get('mid_ratio', 0.5),
                momentum=phase_cfg.get('phase_momentum', 0.01),
                device=self.device,
            )

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
        if self.grad_enabled:
            self.logger.info(
                f"Gradient learner enabled: hidden_dim={self.grad_cfg.get('hidden_dim', 64)}, "
                f"sample_pixels={self.grad_sample_pixels}, alpha={self.grad_alpha}, "
                f"eta={self.grad_eta}, lambda_fit={self.grad_lambda}, warmup={self.grad_warmup}")
        else:
            self.logger.info("Gradient learner disabled")

    def _update_phase_bank_from_feature(self, feature, label):
        if not self.phase_replay_enabled or self.phase_bank is None:
            return

        with torch.no_grad():
            small_label = F.interpolate(
                label.unsqueeze(1).float(), size=feature.shape[-2:], mode="nearest"
            ).squeeze(1).long()

            for cls_idx in torch.unique(small_label):
                if cls_idx <= 0:
                    continue
                mask = (small_label == int(cls_idx)).float().unsqueeze(1)
                if mask.sum() == 0:
                    continue
                masked_feature = feature * mask
                self.phase_bank.update_from_masked_feature(masked_feature, int(cls_idx))

    def _train_gradient_learner(self, logit, label):
        """Train gradient learner on trusted foreground pixels only."""
        if self.grad_learner is None or self.grad_optimizer is None:
            return None

        with torch.no_grad():
            mask_fg = (label != 0) & (label != 255)
            flat_mask = mask_fg.view(-1)
            valid_indices = flat_mask.nonzero(as_tuple=False).squeeze(1)

        if valid_indices.numel() == 0:
            return None

        sample_num = min(self.grad_sample_pixels, valid_indices.numel())
        perm = torch.randperm(valid_indices.numel(), device=valid_indices.device)[:sample_num]
        chosen = valid_indices[perm]

        logit_flat = logit.permute(0, 2, 3, 1).reshape(-1, logit.shape[1])
        label_flat = label.view(-1)

        z_fg = logit_flat[chosen].detach().requires_grad_(True)
        y_fg = label_flat[chosen] - 1  # shift to zero-based class index

        fg_loss = F.cross_entropy(z_fg, y_fg, reduction='mean')
        g_true = torch.autograd.grad(fg_loss, z_fg, retain_graph=False)[0]

        tau = g_true.norm(p=2, dim=1, keepdim=True)

        pred_grad = self.grad_learner(z_fg.detach())
        pred_norm = pred_grad.norm(p=2, dim=1, keepdim=True)
        scaled_grad = self.grad_alpha * tau * pred_grad / (pred_norm + self.grad_epsilon)

        z_prime = z_fg.detach() - self.grad_eta * scaled_grad
        fit_loss = self.grad_lambda * F.cross_entropy(z_prime, y_fg, reduction='mean')

        self.grad_optimizer.zero_grad(set_to_none=True)
        fit_loss.backward()
        self.grad_optimizer.step()

        return fit_loss.item()

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
            opt_stepped = False
            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                ret_intermediate = self.phase_replay_enabled
                logit, features, _ = self.model(data['image'], ret_intermediate=ret_intermediate)

                loss_mbce = self.BCELoss(
                    logit[:, -self.n_new_classes:],  # [N, |Ct|, H, W]
                    data['label'],                # [N, H, W]
                ).mean(dim=[0, 2, 3])  # [|Ct|]

                loss = self.mbce_weight * loss_mbce.sum()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            opt_stepped = True
            self.scaler.update()

            if self.phase_replay_enabled:
                self._update_phase_bank_from_feature(features[-1], data['label'])

            self.optimizer.zero_grad(set_to_none=True)

            if self.grad_enabled and (epoch > self.grad_warmup):
                grad_loss = self._train_gradient_learner(logit.detach(), data['label'])
            else:
                grad_loss = None

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_mbce', loss_mbce.sum().item())
            if grad_loss is not None:
                self.train_metrics.update('loss_grad', grad_loss)

            # Get First lr
            if batch_idx == 0:
                self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch - 1)
                self.logger.info(f"lr[0]: {self.optimizer.param_groups[0]['lr']:.6f} / lr[1]: {self.optimizer.param_groups[1]['lr']:.6f} / lr[2]: {self.optimizer.param_groups[2]['lr']:.6f}")

            if self.lr_scheduler is not None and opt_stepped:
                self.lr_scheduler.step()

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

    def _forward_loss_pass(self, data, update_phase_bank=False, return_logit=False):
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

            phase_replay_active = (
                self.phase_replay_enabled
                and self.phase_bank is not None
                and features_old is not None
            )

            def _legacy_fake_feature(cls_idx, num_samples: int):
                base = self.prev_prototypes[cls_idx].reshape(1, -1, 1, 1)
                per_cls_fake = base.repeat(1, 1, num_samples, 1)
                noise = torch.randn_like(per_cls_fake) * self.prev_noise[cls_idx].reshape(1, -1, 1, 1)
                per_cls_fake = per_cls_fake + noise
                rand_norm = (
                    torch.randn_like(per_cls_fake) * self.prev_norm[1, cls_idx].reshape(1, 1, 1, 1)
                    + self.prev_norm[0, cls_idx].reshape(1, 1, 1, 1)
                )
                return per_cls_fake * rand_norm

            def _phase_fake_feature_from_pred(class_id: int, num_phase: int, pred_resized: torch.Tensor,
                                              features_old_top: torch.Tensor, mid_slices):
                channels = self.prev_prototypes.shape[1]
                if (
                    num_phase <= 0
                    or pred_resized is None
                    or features_old_top is None
                    or mid_slices is None
                    or not phase_replay_active
                    or not self.phase_bank.has_class(class_id)
                ):
                    return torch.empty(1, channels, 0, 1, device=self.device)

                mask_cls = (pred_resized == class_id).float().unsqueeze(1)
                if mask_cls.sum() == 0:
                    return torch.empty(1, channels, 0, 1, device=self.device)

                masked_feature = features_old_top * mask_cls
                amplitude, phase = decompose_spectrum(masked_feature)
                amp_mid, _ = extract_mid(amplitude, phase, mid_slices)

                phase_mid_proto = self.phase_bank.get_phase(class_id).unsqueeze(0).expand_as(amp_mid)
                amp_repl, phase_repl = replace_mid(amplitude, phase, amp_mid, phase_mid_proto, mid_slices)
                recon_feature = reconstruct_feature(amp_repl, phase_repl, enforce_hermitian=True).detach() * mask_cls

                flat_feature = recon_feature.permute(0, 2, 3, 1).reshape(-1, recon_feature.shape[1])
                flat_mask = mask_cls.view(-1) > 0
                selected = flat_feature[flat_mask]

                if selected.shape[0] == 0:
                    return torch.empty(1, channels, 0, 1, device=self.device)

                if selected.shape[0] >= num_phase:
                    idx = torch.randperm(selected.shape[0], device=selected.device)[:num_phase]
                else:
                    idx = torch.randint(0, selected.shape[0], (num_phase,), device=selected.device)
                chosen = selected[idx]
                return chosen.transpose(0, 1).unsqueeze(0).unsqueeze(-1)

            def _phase_fake_feature_from_prototype_only(class_id: int, num_phase: int, full_shape_hw, mid_slices):
                channels = self.prev_prototypes.shape[1]
                if (
                    num_phase <= 0
                    or mid_slices is None
                    or full_shape_hw is None
                    or not phase_replay_active
                    or not self.phase_bank.has_class(class_id)
                ):
                    return torch.empty(1, channels, 0, 1, device=self.device)

                h, w = full_shape_hw
                hs, ws = mid_slices
                phase_mid_proto = self.phase_bank.get_phase(class_id)
                amp_mean_mid, amp_std_mid = self.phase_bank.get_amp_stats(class_id)

                min_norm = self.phase_proto_sampling_min_norm
                max_trials = self.phase_proto_sampling_max_trials
                remaining = num_phase
                vectors = []

                for _ in range(max_trials):
                    if remaining <= 0:
                        break
                    amp_mid_sample = amp_mean_mid + self.phase_proto_amp_scale * amp_std_mid * torch.randn_like(amp_std_mid)
                    amplitude_full = torch.zeros((1, channels, h, w), device=self.device, dtype=amp_mid_sample.dtype)
                    phase_full = torch.zeros_like(amplitude_full)
                    amplitude_full[..., hs, ws] = amp_mid_sample.unsqueeze(0)
                    phase_full[..., hs, ws] = phase_mid_proto.unsqueeze(0)

                    proto_feature = reconstruct_feature(amplitude_full, phase_full, enforce_hermitian=True).detach()
                    flat_feature = proto_feature.permute(0, 2, 3, 1).reshape(-1, channels)
                    norms = flat_feature.norm(p=2, dim=1)
                    valid_idx = torch.nonzero(norms > min_norm, as_tuple=False).squeeze(1)
                    if valid_idx.numel() == 0:
                        continue

                    sample_num = min(remaining, valid_idx.numel())
                    perm = torch.randperm(valid_idx.numel(), device=self.device)[:sample_num]
                    chosen_idx = valid_idx[perm]
                    chosen_vecs = flat_feature[chosen_idx]
                    vectors.append(chosen_vecs)
                    remaining -= chosen_vecs.shape[0]

                if not vectors:
                    return torch.empty(1, channels, 0, 1, device=self.device)

                stacked = torch.cat(vectors, dim=0)
                if stacked.shape[0] > num_phase:
                    perm = torch.randperm(stacked.shape[0], device=self.device)[:num_phase]
                    stacked = stacked[perm]

                return stacked.transpose(0, 1).unsqueeze(0).unsqueeze(-1)

            # --- Direction C: feature-map replay helpers ---
            def _unwrap_model(m):
                return m.module if isinstance(m, (nn.DataParallel, DDP)) else m

            def _phase_reconstruct_map_from_pred(
                class_id: int,
                pred_resized: torch.Tensor,
                features_old_top: torch.Tensor,
                mid_slices,
            ):
                """Reconstruct a feature map using old-model predicted region.

                Returns:
                    recon_map: Tensor[1, C, h, w] or None
                    mask_cls:  Tensor[1, 1, h, w] or None
                """
                if (
                    pred_resized is None
                    or features_old_top is None
                    or mid_slices is None
                    or not phase_replay_active
                    or not self.phase_bank.has_class(class_id)
                ):
                    return None, None

                mask_cls = (pred_resized == class_id).float().unsqueeze(1)
                if mask_cls.sum() == 0:
                    return None, None

                masked_feature = features_old_top * mask_cls
                amplitude, phase = decompose_spectrum(masked_feature)
                amp_mid, _ = extract_mid(amplitude, phase, mid_slices)
                phase_mid_proto = self.phase_bank.get_phase(class_id).unsqueeze(0).expand_as(amp_mid)
                amp_repl, phase_repl = replace_mid(amplitude, phase, amp_mid, phase_mid_proto, mid_slices)

                recon_map = reconstruct_feature(amp_repl, phase_repl, enforce_hermitian=True).detach()
                recon_map = recon_map * mask_cls  # keep only the class region; outside is 0
                return recon_map, mask_cls

            def _phase_reconstruct_map_from_prototype_only(class_id: int, full_shape_hw, mid_slices):
                """Reconstruct a full feature map from phase/amplitude prototypes only."""
                if (
                    full_shape_hw is None
                    or mid_slices is None
                    or not phase_replay_active
                    or not self.phase_bank.has_class(class_id)
                ):
                    return None, None

                channels = self.prev_prototypes.shape[1]
                h, w = full_shape_hw
                hs, ws = mid_slices

                phase_mid_proto = self.phase_bank.get_phase(class_id)
                amp_mean_mid, amp_std_mid = self.phase_bank.get_amp_stats(class_id)

                min_norm = self.phase_proto_sampling_min_norm
                max_trials = self.phase_proto_sampling_max_trials
                for _ in range(max_trials):
                    amp_mid_sample = amp_mean_mid + self.phase_proto_amp_scale * amp_std_mid * torch.randn_like(amp_std_mid)

                    amplitude_full = torch.zeros((1, channels, h, w), device=self.device, dtype=amp_mid_sample.dtype)
                    phase_full = torch.zeros_like(amplitude_full)
                    amplitude_full[..., hs, ws] = amp_mid_sample.unsqueeze(0)
                    phase_full[..., hs, ws] = phase_mid_proto.unsqueeze(0)

                    proto_map = reconstruct_feature(amplitude_full, phase_full, enforce_hermitian=True).detach()

                    # Ensure the map isn't degenerate (all~0)
                    mean_norm = proto_map.flatten(2).norm(p=2, dim=1).mean()
                    if float(mean_norm) > float(min_norm):
                        return proto_map, None

                return None, None

            mid_slices = None
            pred_small = None
            features_old_top = None
            if phase_replay_active:
                _, _, h, w = features_old[-1].shape
                mid_slices = get_mid_slices(h, w, self.phase_bank.mid_ratio)
                pred_small = (
                    F.interpolate(pred.unsqueeze(1).float(), size=(h, w), mode="nearest")
                    .squeeze(1)
                    .long()
                )
                features_old_top = features_old[-1]

            fake_features_legacy = []
            fake_features_phase = []
            channels = self.prev_prototypes.shape[1]
            empty_feature = torch.empty(1, channels, 0, 1, device=self.device)

            for cls in range(0, self.per_iter_prev_number.shape[0]):
                num_samples = int(self.per_iter_prev_number[cls].item())
                class_id = self.task_info['old_class'][cls] if 'old_class' in self.task_info else cls + 1

                if num_samples == 0:
                    fake_features_legacy.append(empty_feature)
                    fake_features_phase.append(empty_feature)
                    continue

                # 旧版伪特征始终生成
                per_cls_legacy = _legacy_fake_feature(cls, num_samples)
                fake_features_legacy.append(per_cls_legacy)

                # 相位伪特征尝试生成
                if phase_replay_active and self.phase_bank.has_class(class_id):
                    phase_k = int(round(num_samples * self.phase_ratio))
                    if phase_k > 0:
                        per_cls_phase_list = []
                        phase_from_pred = _phase_fake_feature_from_pred(
                            class_id, phase_k, pred_small, features_old_top, mid_slices
                        )
                        if phase_from_pred.shape[2] > 0:
                            per_cls_phase_list.append(phase_from_pred)

                        remaining = phase_k - sum(item.shape[2] for item in per_cls_phase_list)
                        if remaining > 0:
                            phase_from_proto = _phase_fake_feature_from_prototype_only(
                                class_id, remaining, (h, w) if features_old_top is not None else None, mid_slices
                            )
                            if phase_from_proto.shape[2] > 0:
                                per_cls_phase_list.append(phase_from_proto)

                        if per_cls_phase_list:
                            fake_features_phase.append(torch.cat(per_cls_phase_list, dim=2))
                        else:
                            fake_features_phase.append(empty_feature)
                    else:
                        fake_features_phase.append(empty_feature)
                else:
                    fake_features_phase.append(empty_feature)

            fake_legacy = torch.cat(fake_features_legacy, dim=2)
            fake_phase = torch.cat(fake_features_phase, dim=2)
            fake_features = torch.cat([fake_legacy, fake_phase], dim=2)

            n_legacy = fake_legacy.shape[2]
            n_phase = fake_phase.shape[2]
            n_total_fake = fake_features.shape[2]

            fake_label_legacy = torch.zeros(1, n_legacy, 1, requires_grad=False).to(self.device)
            fake_label_phase = torch.zeros(1, n_phase, 1, requires_grad=False).to(self.device)
            fake_label = torch.zeros(1, n_total_fake, 1, requires_grad=False).to(self.device)

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

            zeros_like_ori = torch.zeros_like(loss_mbce_ori)
            loss_mbce_fake_legacy = zeros_like_ori
            loss_mbce_fake_phase = zeros_like_ori

            if n_legacy > 0:
                logits_fake_legacy = logits_for_fake[:, :, :n_legacy, :]
                loss_mbce_fake_legacy = self.BCELoss_fake(
                    logits_fake_legacy[:, -self.n_new_classes:], fake_label_legacy
                ).mean(dim=[0, 2, 3])

            if n_phase > 0:
                logits_fake_phase = logits_for_fake[:, :, n_legacy:n_total_fake, :]
                loss_mbce_fake_phase = self.BCELoss_fake(
                    logits_fake_phase[:, -self.n_new_classes:], fake_label_phase
                ).mean(dim=[0, 2, 3])

            stride_num = features[-1].shape[0] * features[-1].shape[2] * features[-1].shape[3]
            weight_extra_bg = self.extra_bg_ratio * region_bg.sum() / stride_num
            weight_fake_legacy = n_legacy / stride_num
            weight_fake_phase = n_phase / stride_num

            loss_mbce = loss_mbce_ori
            loss_mbce = loss_mbce + loss_mbce_fake_legacy * weight_fake_legacy
            loss_mbce = loss_mbce + loss_mbce_fake_phase * weight_fake_phase * self.phase_loss_weight
            loss_mbce = loss_mbce + loss_mbce_extra_bg * weight_extra_bg

            denom = 1 + weight_extra_bg + weight_fake_legacy + weight_fake_phase * self.phase_loss_weight
            loss_mbce = loss_mbce / denom

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

            # --- Direction C: feature-map replay losses (optional) ---
            loss_map_total = torch.tensor(0.0, device=self.device)
            loss_map_distill = torch.tensor(0.0, device=self.device)
            loss_map_new0 = torch.tensor(0.0, device=self.device)
            loss_map_grad = torch.tensor(0.0, device=self.device)
            loss_map_local = torch.tensor(0.0, device=self.device)
            num_map_replay = torch.tensor(0.0, device=self.device)

            if (
                self.map_replay_enabled
                and self.map_replay_num_maps > 0
                and phase_replay_active
                and (self.model_old is not None)
            ):
                student = _unwrap_model(self.model)
                teacher = _unwrap_model(self.model_old)

                # Candidate old classes (only those with replay quota and phase prototypes)
                candidates = []
                for cls in range(0, self.per_iter_prev_number.shape[0]):
                    if int(self.per_iter_prev_number[cls].item()) <= 0:
                        continue
                    class_id = self.task_info['old_class'][cls] if 'old_class' in self.task_info else cls + 1
                    if self.phase_bank.has_class(class_id):
                        candidates.append(int(class_id))

                if candidates:
                    num_pick = min(self.map_replay_num_maps, len(candidates))
                    perm = torch.randperm(len(candidates), device=self.device)[:num_pick]
                    picked = [candidates[int(i)] for i in perm]

                    eps = self.map_replay_eps
                    conf_thresh = self.map_replay_teacher_conf

                    for class_id in picked:
                        # Prefer pred-based reconstruction when possible.
                        proto_map, proto_mask = None, None
                        if self.map_replay_use_pred_first:
                            proto_map, proto_mask = _phase_reconstruct_map_from_pred(
                                class_id, pred_small, features_old_top, mid_slices
                            )
                        if proto_map is None:
                            proto_map, proto_mask = _phase_reconstruct_map_from_prototype_only(
                                class_id, (h, w) if features_old_top is not None else None, mid_slices
                            )
                        if proto_map is None:
                            continue

                        logits_s_map = student.forward_from_top_feature(proto_map)
                        with torch.no_grad():
                            logits_t_map = teacher.forward_from_top_feature(proto_map)

                        c_prev = logits_t_map.shape[1]
                        if c_prev <= 0:
                            continue

                        logits_s_old = logits_s_map[:, :c_prev]
                        p_t_old = torch.sigmoid(logits_t_map)

                        # Teacher confidence mask (optional)
                        if conf_thresh > 0:
                            conf = p_t_old.max(dim=1, keepdim=True).values  # [1,1,h,w]
                            mask = (conf > conf_thresh).float()
                        else:
                            mask = torch.ones_like(p_t_old[:, :1])

                        # Distill old-class responses (pixel-wise)
                        distill_raw = F.binary_cross_entropy_with_logits(logits_s_old, p_t_old, reduction='none')
                        loss_d = (distill_raw * mask).sum() / (mask.sum() * c_prev + eps)

                        # Suppress new-class channels (pixel-wise)
                        if self.n_new_classes > 0:
                            logits_s_new = logits_s_map[:, -self.n_new_classes:]
                            zeros = torch.zeros_like(logits_s_new)
                            new0_raw = F.binary_cross_entropy_with_logits(logits_s_new, zeros, reduction='none')
                            loss_n0 = (new0_raw * mask).sum() / (mask.sum() * self.n_new_classes + eps)
                        else:
                            loss_n0 = torch.tensor(0.0, device=self.device)

                        # Gradient consistency on probability maps (optional)
                        if self.map_replay_w_grad > 0:
                            p_s_old = torch.sigmoid(logits_s_old)
                            dx_s = p_s_old[..., :, 1:] - p_s_old[..., :, :-1]
                            dx_t = p_t_old[..., :, 1:] - p_t_old[..., :, :-1]
                            dy_s = p_s_old[..., 1:, :] - p_s_old[..., :-1, :]
                            dy_t = p_t_old[..., 1:, :] - p_t_old[..., :-1, :]
                            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
                            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
                            loss_g = (dx_s - dx_t).abs()
                            loss_g = (loss_g * mask_x).sum() / (mask_x.sum() * c_prev + eps)
                            loss_g2 = (dy_s - dy_t).abs()
                            loss_g2 = (loss_g2 * mask_y).sum() / (mask_y.sum() * c_prev + eps)
                            loss_g = loss_g + loss_g2
                        else:
                            loss_g = torch.tensor(0.0, device=self.device)

                        # Local similarity consistency (optional)
                        if self.map_replay_w_local > 0:
                            p_s_old = torch.sigmoid(logits_s_old)

                            def _cos(a, b):
                                num = (a * b).sum(dim=1)
                                den = (a.norm(p=2, dim=1) * b.norm(p=2, dim=1)).clamp_min(eps)
                                return num / den

                            sim_s_x = _cos(p_s_old[..., :, 1:], p_s_old[..., :, :-1])
                            sim_t_x = _cos(p_t_old[..., :, 1:], p_t_old[..., :, :-1])
                            sim_s_y = _cos(p_s_old[..., 1:, :], p_s_old[..., :-1, :])
                            sim_t_y = _cos(p_t_old[..., 1:, :], p_t_old[..., :-1, :])
                            mask_x2 = mask[..., :, 1:].squeeze(1) * mask[..., :, :-1].squeeze(1)
                            mask_y2 = mask[..., 1:, :].squeeze(1) * mask[..., :-1, :].squeeze(1)
                            loss_lx = (sim_s_x - sim_t_x).abs()
                            loss_lx = (loss_lx * mask_x2).sum() / (mask_x2.sum() + eps)
                            loss_ly = (sim_s_y - sim_t_y).abs()
                            loss_ly = (loss_ly * mask_y2).sum() / (mask_y2.sum() + eps)
                            loss_l = loss_lx + loss_ly
                        else:
                            loss_l = torch.tensor(0.0, device=self.device)

                        loss_map_distill = loss_map_distill + loss_d
                        loss_map_new0 = loss_map_new0 + loss_n0
                        loss_map_grad = loss_map_grad + loss_g
                        loss_map_local = loss_map_local + loss_l
                        num_map_replay = num_map_replay + 1.0

                    if num_map_replay.item() > 0:
                        loss_map_distill = loss_map_distill / num_map_replay
                        loss_map_new0 = loss_map_new0 / num_map_replay
                        loss_map_grad = loss_map_grad / num_map_replay
                        loss_map_local = loss_map_local / num_map_replay
                        loss_map_total = (
                            self.map_replay_w_distill * loss_map_distill
                            + self.map_replay_w_new0 * loss_map_new0
                            + self.map_replay_w_grad * loss_map_grad
                            + self.map_replay_w_local * loss_map_local
                        )

        if update_phase_bank:
            self._update_phase_bank_from_feature(features[-1], data['label'])

        loss_dict = {
            'loss_mbce': loss_mbce,
            'loss_mbce_distill': loss_mbce_distill,
            'loss_pkd': loss_pkd,
            'loss_cont': loss_cont,
            'loss_map_total': loss_map_total,
            'loss_map_distill': loss_map_distill,
            'loss_map_new0': loss_map_new0,
            'loss_map_grad': loss_map_grad,
            'loss_map_local': loss_map_local,
            'num_map_replay': num_map_replay,
            'consistency_ratio': consistency_ratio,
            'loss_mbce_fake_legacy': loss_mbce_fake_legacy,
            'loss_mbce_fake_phase': loss_mbce_fake_phase,
            'num_fake_legacy': torch.tensor(float(n_legacy), device=self.device),
            'num_fake_phase': torch.tensor(float(n_phase), device=self.device),
        }

        if return_logit:
            return loss_dict, logit.detach()
        return loss_dict

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

    def _test(self, epoch=None, return_metrics=False):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        log = {}
        raw_metrics = {} if return_metrics else None
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
                met_result = met()
                if return_metrics:
                    raw_metrics[met.__name__] = met_result

                if epoch is not None:
                    if len(met_result.keys()) > 2:
                        self.test_metrics.update(met.__name__, [met_result['old'], met_result['new'], met_result['harmonic']], 'old', 'new', 'harmonic', n=1)
                    else:
                        self.test_metrics.update(met.__name__, [met_result['overall']], 'overall', n=1)

                if 'old' in met_result.keys():
                    log.update({met.__name__ + '_old': f"{met_result['old']:.2f}"})
                if 'new' in met_result.keys():
                    log.update({met.__name__ + '_new': met_result['new']})
                if 'harmonic' in met_result.keys():
                    log.update({met.__name__ + '_harmonic': met_result['harmonic']})
                if 'overall' in met_result.keys():
                    log.update({met.__name__ + '_overall': f"{met_result['overall']:.2f}"})
                if 'by_class' in met_result.keys():
                    by_class_str = '\n'
                    for i in range(len(met_result['by_class'])):
                        if i in self.evaluator_test.new_classes_idx:
                            by_class_str = by_class_str + f"{i:2d} *{VOC[i]} {met_result['by_class'][i]:.2f}\n"
                        else:
                            by_class_str = by_class_str + f"{i:2d}  {VOC[i]} {met_result['by_class'][i]:.2f}\n"
                    log.update({met.__name__ + '_by_class': by_class_str})
        if return_metrics:
            return log, raw_metrics
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
            'loss_old_step', 'consistency_ratio', 'loss_grad',
            'loss_fake_legacy', 'loss_fake_phase', 'num_fake_legacy', 'num_fake_phase',
            'loss_map', 'loss_map_distill', 'loss_map_new0', 'loss_map_grad', 'loss_map_local', 'num_map_replay',
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

        phase_cfg = self.config['hyperparameter'].get('phase_replay', {})
        self.phase_replay_enabled = phase_cfg.get('enabled', False)
        if self.phase_replay_enabled:
            self.phase_bank = PhasePrototypeBank(
                mid_ratio=phase_cfg.get('mid_ratio', 0.5),
                momentum=phase_cfg.get('phase_momentum', 0.01),
                device=self.device,
            )
            prev_phase_state = prev_info.get('phase_bank', None)
            if prev_phase_state is not None:
                self.phase_bank.load_state_dict(prev_phase_state, map_location=self.device)
        else:
            self.phase_bank = None

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
        manual_proto = config.config.get('prev_prototypes_path', None)
        if manual_proto is not None:
            return Path(manual_proto)

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
            return_logit = self.grad_enabled and (epoch > self.grad_warmup)
            loss_out = self._forward_loss_pass(
                data, update_phase_bank=self.phase_replay_enabled, return_logit=return_logit)
            if return_logit:
                loss_dict, grad_logit = loss_out
            else:
                loss_dict, grad_logit = loss_out, None

            distill_weight = self.mbce_distill_weight if self.enable_mbce_distill else 0
            loss_main = self.mbce_weight * loss_dict['loss_mbce'].sum()
            loss_old = distill_weight * loss_dict['loss_mbce_distill'].sum() \
                       + self.config['hyperparameter']['pkd'] * loss_dict['loss_pkd'].sum() \
                       + self.config['hyperparameter']['cont'] * loss_dict['loss_cont'] \
                       + self.map_replay_weight * loss_dict['loss_map_total']

            opt_stepped = False
            if not self.use_separate_old_update:
                loss = loss_main + loss_old
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                opt_stepped = True
                self.scaler.update()
            else:
                # 主监督更新
                self.scaler.scale(loss_main).backward()
                self.scaler.step(self.optimizer)
                opt_stepped = True
                self.scaler.update()

                # 独立的旧类伪梯度/蒸馏更新
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict_old = self._forward_loss_pass(
                    data, update_phase_bank=False, return_logit=False)
                loss_old_step = distill_weight * loss_dict_old['loss_mbce_distill'].sum() \
                                 + self.config['hyperparameter']['pkd'] * loss_dict_old['loss_pkd'].sum() \
                                 + self.config['hyperparameter']['cont'] * loss_dict_old['loss_cont'] \
                                 + self.map_replay_weight * loss_dict_old['loss_map_total']

                if loss_old_step.requires_grad:
                    loss_old_scaled = self.pseudo_grad_scale * loss_old_step
                    self.scaler.scale(loss_old_scaled).backward()
                    self.scaler.step(self.optimizer)
                    opt_stepped = True
                    self.scaler.update()
                loss_old = loss_old_step

            grad_loss = None
            if self.grad_enabled and (epoch > self.grad_warmup) and (grad_logit is not None):
                grad_loss = self._train_gradient_learner(grad_logit, data['label'])

            # 统一的 lr_scheduler 步进（仅在本轮确实执行了 optimizer.step 时）
            if self.lr_scheduler is not None and opt_stepped:
                self.lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', (loss_main + loss_old).item())
            self.train_metrics.update('loss_mbce', loss_dict['loss_mbce'].sum().item() * self.mbce_weight)
            self.train_metrics.update('loss_bce_distill',
                                      loss_dict['loss_mbce_distill'].sum().item() * distill_weight)
            self.train_metrics.update('loss_pkd', loss_dict['loss_pkd'].sum().item() * self.config['hyperparameter']['pkd'])
            self.train_metrics.update('loss_cont', loss_dict['loss_cont'].item() * self.config['hyperparameter']['cont'])
            self.train_metrics.update('loss_map', loss_dict['loss_map_total'].item() * self.map_replay_weight)
            self.train_metrics.update('loss_map_distill', loss_dict['loss_map_distill'].item())
            self.train_metrics.update('loss_map_new0', loss_dict['loss_map_new0'].item())
            self.train_metrics.update('loss_map_grad', loss_dict['loss_map_grad'].item())
            self.train_metrics.update('loss_map_local', loss_dict['loss_map_local'].item())
            self.train_metrics.update('num_map_replay', loss_dict['num_map_replay'].item())
            self.train_metrics.update('loss_fake_legacy', loss_dict['loss_mbce_fake_legacy'].sum().item())
            self.train_metrics.update('loss_fake_phase', loss_dict['loss_mbce_fake_phase'].sum().item())
            self.train_metrics.update('num_fake_legacy', loss_dict['num_fake_legacy'].item())
            self.train_metrics.update('num_fake_phase', loss_dict['num_fake_phase'].item())
            self.train_metrics.update('loss_old_step', loss_old.item() if torch.is_tensor(loss_old) else float(loss_old))
            self.train_metrics.update('consistency_ratio', loss_dict['consistency_ratio'].item())
            if grad_loss is not None:
                self.train_metrics.update('loss_grad', grad_loss)

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
