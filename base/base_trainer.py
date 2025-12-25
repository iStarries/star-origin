import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from models.modules.phase_proto import PhasePrototypeBank


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, gpu):
        self.config = config
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['epochs'] if cfg_trainer['save_period'] == -1 else cfg_trainer['save_period']
        # self.save_period = 1
        self.validation_period = cfg_trainer['validation_period'] if cfg_trainer['validation_period'] == -1 else cfg_trainer['validation_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.reset_best_mnt = cfg_trainer['reset_best_mnt']
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        if logger is None:
            self.logger = config.get_logger('trainer', cfg_trainer['verbosity'])
        else:
            self.logger = logger
            # setup visualization writer instance
            if self.rank == 0:
                self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])
            else:
                self.writer = TensorboardWriter(config.log_dir, self.logger, False)
        
        if gpu is None:
            # setup GPU device if available, move model into configured device
            self.device, self.device_ids = self._prepare_device(config['n_gpu'])
        else:
            self.device = gpu
            self.device_ids = None

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        self.enable_test_validation = self.config.config.get('validate', False)
        self.test_validation_window = 25
        self.test_best_miou = -inf
        self.test_best_path = None
        self.log_file_path = self.config.log_dir / "info.log"
        self._init_phase_replay_config()

        self.phase_replay_cfg = self.config.config.get('phase_replay', {})
        self.phase_replay_enabled = bool(self.phase_replay_cfg.get('enabled', False))
        self.ppb = None


        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

    def _init_phase_replay_config(self):
        cfg_dict = getattr(self.config, "config", self.config)
        self.phase_replay_cfg = cfg_dict.get('phase_replay', {}) if isinstance(cfg_dict, dict) else {}
        self.phase_replay_enabled = self.phase_replay_cfg.get('enabled', False)
        self.ppb = None
        self.ppb_background_id = 0
        self.ppb_ignore_id = 255
        self.train_id_classes = []
        self.train_id_old_classes = []
        self.train_id_new_classes = []
        self.train_id_new_classes_set = set()
        self.label_to_train_id = {}
        self._label_map = None

    def _init_train_id_mapping(self, task_info):
        train_id_classes = list(task_info.get('old_class', [])) + list(task_info.get('new_class', []))
        self.train_id_classes = train_id_classes
        self.label_to_train_id = {label: idx for idx, label in enumerate(train_id_classes)}
        n_old = len(task_info.get('old_class', []))
        n_total = len(train_id_classes)
        self.train_id_old_classes = list(range(n_old))
        self.train_id_new_classes = list(range(n_old, n_total))
        self.train_id_new_classes_set = set(self.train_id_new_classes)
        max_label = max([self.ppb_ignore_id] + train_id_classes + [self.ppb_background_id])
        self._label_map = torch.full((max_label + 1,), -1, dtype=torch.long)
        for label, train_id in self.label_to_train_id.items():
            if 0 <= label < self._label_map.numel():
                self._label_map[label] = train_id

    def _map_label_to_train_id(self, label_tensor):
        if self._label_map is None:
            return label_tensor
        max_label = int(label_tensor.max().item())
        if max_label >= self._label_map.numel():
            new_map = torch.full((max_label + 1,), -1, dtype=torch.long)
            new_map[: self._label_map.numel()] = self._label_map
            for label, train_id in self.label_to_train_id.items():
                if 0 <= label < new_map.numel():
                    new_map[label] = train_id
            self._label_map = new_map
        return self._label_map.to(label_tensor.device)[label_tensor]

    def _dist_barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _forward_head(self, feature_tensor):
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self.model.module.forward_class_prediction(feature_tensor)
        return self.model.forward_class_prediction(feature_tensor)

    def _maybe_init_ppb(self, logit, features):
        """
        1) phase_replay 未启用：不做任何事
        2) phase_replay 启用且 ppb 已存在：强校验 ppb.num_classes == head_out_channels
        3) phase_replay 启用且 ppb 不存在：按当前 step 的 train_id 类空间初始化（并校验与 head_out_channels 一致）
        """
        if not self.phase_replay_enabled:
            return

        head_out = int(logit.shape[1])

        # ppb 已存在：每个 step 的第一批就强校验，避免静默错配
        if self.ppb is not None:
            if int(self.ppb.num_classes) != head_out:
                raise RuntimeError(
                    f"[PHASE-PPB] PPB num_classes mismatch: ppb.num_classes={int(self.ppb.num_classes)} "
                    f"!= head_out_channels={head_out}. This indicates wrong PPB load/expand or head channels mismatch."
                )
            return

        # ppb 不存在：需要初始化
        if not isinstance(features, (list, tuple)) or len(features) == 0:
            raise RuntimeError("[PHASE-PPB] ret_intermediate=True required but features is empty.")

        feature_tensor = features[-1]
        feat_shape = tuple(feature_tensor.shape[1:])  # (C,H,W)

        # 最稳：PPB 的 class_id 空间必须等于 train_id 空间大小
        desired_num = len(getattr(self, "train_id_classes", []))
        if desired_num <= 0:
            raise RuntimeError(
                "[PHASE-PPB] train_id_classes is empty; _init_train_id_mapping() must run before PPB init.")

        # 强校验：train_id 空间大小必须等于 head 输出通道数
        if desired_num != head_out:
            raise RuntimeError(
                f"[PHASE-PPB] train_id_classes length mismatch head_out_channels: "
                f"len(train_id_classes)={desired_num} != head_out_channels={head_out}. "
                f"Check label remap / head expand logic."
            )

        self.ppb = PhasePrototypeBank(
            num_classes=desired_num,
            feat_shape=feat_shape,
            r_low_ratio=self.phase_replay_cfg.get('r_low_ratio', 0.2),
            r_high_ratio=self.phase_replay_cfg.get('r_high_ratio', 0.6),
            ema_beta=self.phase_replay_cfg.get('ema_beta', 0.01),
            use_cos_sin=self.phase_replay_cfg.get('use_cos_sin', True),
            normalize_syn=self.phase_replay_cfg.get('normalize_syn', False),
            replay_detach_ref=self.phase_replay_cfg.get('replay_detach_ref', True),
        )
        self.ppb.to(self.device)

    def _load_phase_ppb(self, config):
        """
        从 step_{t-1} 读取 phase_ppb.pth，然后：
        - 用“当前 step 的总 train_id 类数”作为 PPB num_classes（必须扩容）
        - 将旧 PPB 的 [0:prev_num] 统计拷贝进新 PPB
        - 强校验 mid_mask 与 (C, n_mid) 等维度一致，否则直接报错
        """
        if not self.phase_replay_enabled:
            return

        step = config['data_loader']['args']['task']['step']
        if step <= 0:
            return

        cfg_dict = getattr(config, "config", config)
        ppb_filename = "phase_ppb.pth"
        if isinstance(cfg_dict, dict):
            ppb_filename = cfg_dict.get('phase_replay', {}).get('ppb_filename', ppb_filename)

        prev_dir = config.save_dir.parent / f"step_{step - 1}"
        ppb_path = prev_dir / ppb_filename
        if not ppb_path.exists():
            raise RuntimeError(f"[PHASE-PPB] Expected PPB file not found: {ppb_path}")

        payload = torch.load(ppb_path, map_location="cpu")
        meta = payload.get("meta", {})

        prev_num = meta.get("num_classes")
        feat_shape = tuple(meta.get("feat_shape", ()))
        if prev_num is None or len(feat_shape) != 3:
            raise RuntimeError(f"[PHASE-PPB] Invalid PPB meta in {ppb_path}: meta={meta}")

        prev_num = int(prev_num)

        # 当前 step 的 train_id 总类数（old+new），PPB 必须匹配它
        cur_num = len(getattr(self, "train_id_classes", []))
        if cur_num <= 0:
            raise RuntimeError(
                "[PHASE-PPB] train_id_classes is empty; _init_train_id_mapping() must run before PPB load.")

        # 这里是“有异常就解决/直接报错”的策略：
        # prev_num > cur_num：说明加载的 ppb 比当前 head 更大，属于严重错配，直接报错
        if prev_num > cur_num:
            raise RuntimeError(
                f"[PHASE-PPB] Loaded PPB has more classes than current step: prev_num={prev_num} > cur_num={cur_num}. "
                f"Check step directory / head channels / task split."
            )

        # 用当前 step 的 cur_num 构建新 PPB（扩容）
        self.ppb = PhasePrototypeBank(
            num_classes=cur_num,
            feat_shape=feat_shape,
            r_low_ratio=meta.get("r_low_ratio", self.phase_replay_cfg.get('r_low_ratio', 0.2)),
            r_high_ratio=meta.get("r_high_ratio", self.phase_replay_cfg.get('r_high_ratio', 0.6)),
            ema_beta=meta.get("ema_beta", self.phase_replay_cfg.get('ema_beta', 0.01)),
            use_cos_sin=meta.get("use_cos_sin", self.phase_replay_cfg.get('use_cos_sin', True)),
            normalize_syn=self.phase_replay_cfg.get('normalize_syn', False),
            replay_detach_ref=self.phase_replay_cfg.get('replay_detach_ref', True),
        )

        state = payload.get("state_dict", {})
        required = ["mid_mask", "phase_cos", "phase_sin", "amp_mean", "count"]
        missing = [k for k in required if k not in state]
        if missing:
            raise RuntimeError(f"[PHASE-PPB] Missing keys in state_dict: {missing}. File: {ppb_path}")

        # 强校验 mid_mask 一致（否则说明 r_low/r_high 或 feat_shape 不一致）
        if not torch.equal(self.ppb.mid_mask.cpu(), state["mid_mask"].cpu()):
            raise RuntimeError(
                "[PHASE-PPB] mid_mask mismatch between loaded PPB and current PPB init. "
                "This usually means feat_shape or (r_low_ratio, r_high_ratio) changed."
            )

        # 强校验 (C, n_mid) 一致，再做拷贝
        with torch.no_grad():
            for key in ["phase_cos", "phase_sin", "amp_mean"]:
                src = state[key]
                dst = getattr(self.ppb, key)
                if src.shape[0] != prev_num:
                    raise RuntimeError(
                        f"[PHASE-PPB] {key} prev_num mismatch: src.shape[0]={src.shape[0]} != prev_num={prev_num}")
                if src.shape[1:] != dst.shape[1:]:
                    raise RuntimeError(
                        f"[PHASE-PPB] {key} shape mismatch: src.shape[1:]={tuple(src.shape[1:])} != dst.shape[1:]={tuple(dst.shape[1:])}")
                dst[:prev_num].copy_(src)

            src_count = state["count"]
            if src_count.shape[0] != prev_num:
                raise RuntimeError(f"[PHASE-PPB] count prev_num mismatch: {src_count.shape[0]} != {prev_num}")
            self.ppb.count[:prev_num].copy_(src_count)

        self.ppb.to(self.device)

    def save_phase_ppb(self, config):
        if not self.phase_replay_enabled or self.ppb is None:
            return
        cfg_dict = getattr(config, "config", config)
        ppb_filename = "phase_ppb.pth"
        if isinstance(cfg_dict, dict):
            ppb_filename = cfg_dict.get('phase_replay', {}).get('ppb_filename', ppb_filename)
        payload = {
            "version": 1,
            "meta": {
                "num_classes": self.ppb.num_classes,
                "feat_shape": self.ppb.feat_shape,
                "r_low_ratio": self.phase_replay_cfg.get('r_low_ratio', 0.2),
                "r_high_ratio": self.phase_replay_cfg.get('r_high_ratio', 0.6),
                "use_cos_sin": self.phase_replay_cfg.get('use_cos_sin', True),
                "ema_beta": self.phase_replay_cfg.get('ema_beta', 0.01),
                "ref_mode": self.phase_replay_cfg.get('ref_mode', 'batch_mean'),
            },
            "state_dict": self.ppb.state_dict(),
        }
        torch.save(payload, config.save_dir / ppb_filename)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """

        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_flag = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            test_log = None
            if self.enable_test_validation and epoch >= self.epochs - self.test_validation_window + 1:
                test_log = self._test(epoch)
                log.update(**{'test_' + k: v for k, v in test_log.items()})
                if self.rank == 0:
                    test_miou = self._extract_test_miou(test_log)
                    if test_miou is not None and test_miou > self.test_best_miou:
                        self.test_best_miou = test_miou
                        self._save_best_test_checkpoint(epoch, test_miou)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.rank == 0:
                if val_flag and (self.mnt_mode != 'off'):
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        self._save_best_model(epoch)
                        
                    else:
                        not_improved_count += 1

                    if (self.early_stop > 0) and (not_improved_count > self.early_stop):
                        self.logger.info("Validation performance didn\'train_voc.sh improve for {} epochs. "
                                         "Training stops.".format(self.early_stop))
                        break

                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)

                    self.compute_prototypes(self.config)
                    self.compute_noise(self.config)
                    self.save_prototypes(self.config, epoch)
                    self.save_phase_ppb(self.config)

        # close TensorboardX
        self.writer.close()
        if self.rank == 0:
            self._finalize_info_log()

    def _extract_test_miou(self, test_log):
        if not test_log:
            return None
        miou_key = 'Mean_Intersection_over_Union_overall'
        if miou_key not in test_log:
            self.logger.warning(f"Warning: Metric '{miou_key}' is not found in test log.")
            return None
        return float(test_log[miou_key])

    def _save_best_test_checkpoint(self, epoch, miou):
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = self.checkpoint_dir / f'test_best-epoch{epoch}-miou{miou:.2f}.pth'
        if self.test_best_path is not None and self.test_best_path.exists():
            self.test_best_path.unlink()
        torch.save(state, filename)
        self.test_best_path = filename
        self.logger.info(f"Saving current best test checkpoint: {filename} ...")

    def _finalize_info_log(self, miou_from_test=None):
        if self.rank != 0:
            return
        miou = None
        if self.test_best_miou != -inf:
            miou = self.test_best_miou
        elif miou_from_test is not None:
            miou = miou_from_test

        suffix = f"{miou:.2f}" if miou is not None and miou != -inf else "unknown"
        info_log = self.log_file_path
        if not info_log.exists():
            return
        new_name = info_log.with_name(f"info-miou{suffix}.log")
        if info_log == new_name:
            return
        try:
            if new_name.exists():
                new_name.unlink()
            info_log.rename(new_name)
            self.log_file_path = new_name
            self.logger.info(f"Renamed info log to {new_name.name}")
        except OSError as e:
            self.logger.warning(f"Failed to rename info log: {e}")

    def save_prototypes(self, config, epoch):
        save_file = str(config.save_dir) + "/prototypes-epoch{}.pth".format(epoch)

        all_info = {
            "numbers": self.numbers,
            "prototypes": self.prototypes,
            "norm_mean_and_std": self.norm_mean_and_std,
            "noise": self.noise
        }

        torch.save(all_info, save_file)

    def compute_cls_number(self, config):
        self.logger.info("computing number of pixels...")

        number_save_file = str(config.save_dir) + "/numbers_tmp.pth"
        if os.path.exists(number_save_file):
            self.numbers = torch.load(number_save_file)
            return

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        numbers = torch.zeros(n_new_classes + 1).to(self.device)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                # if batch_idx % 10 == 0:
                #     self.logger.info("[" + str(batch_idx) + "/" + str(len(self.train_loader)) + "]")

                small_label = data['label'][:, 8::16, 8::16].to(self.device)
                for i in range(n_new_classes + 1):
                    if i == 0:
                        numbers[i] = numbers[i] + torch.sum(small_label == 0).item()
                        continue
                    numbers[i] = numbers[i] + torch.sum(small_label == i + n_old_classes).item()
                self.progress(self.logger, batch_idx, len(self.train_loader))

        self.numbers = numbers

        torch.save(numbers, number_save_file)

    def compute_prototypes(self, config):
        # step = config['data_loader']['args']['task']['step']
        # prt_save_file = str(str(config.save_dir) + "/prototypes-epoch{}.pth".format(60))
        # norm_save_file = str(str(config.save_dir) + "/norm-epoch{}.pth".format(60))

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        prototypes = torch.zeros(n_new_classes, 256, device='cuda')
        # prototypes = torch.load(prt_save_file).cuda()
        norms = {k: [] for k in range(n_new_classes)}
        norm_mean_and_std = torch.zeros(2, n_new_classes, device='cuda')

        self.logger.info("computing prototypes...")
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)
                prototypes = prototypes + class_region.sum(dim=[0, 3, 4])

                norm = torch.norm(features[-1], p=2, dim=1)
                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    norms[int(cls) - n_old_classes - 1].append(norm[small_label == cls])

                self.progress(self.logger, batch_idx, len(self.train_loader))

            prototypes = F.normalize(prototypes, p=2, dim=1)
            # torch.save(prototypes, prt_save_file)

            if config['data_loader']['args']['task']['step'] == 0:
                self.prototypes = prototypes
            else:
                self.prototypes = torch.cat([self.prev_prototypes, prototypes], dim=0)

            for k in range(n_new_classes):
                norms[k] = torch.cat(norms[k], dim=0)
                norm_mean_and_std[0, k] = norms[k].mean()
                norm_mean_and_std[1, k] = norms[k].std()
            # torch.save(norm_mean_and_std, norm_save_file)

            if config['data_loader']['args']['task']['step'] == 0:
                self.norm_mean_and_std = norm_mean_and_std
            else:
                self.norm_mean_and_std = torch.cat([self.prev_norm, norm_mean_and_std], dim=1)
            
        self.model.train()

    def compute_noise(self, config):
        # step = config['data_loader']['args']['task']['step']
        # prt_save_file = str(str(config.save_dir) + "/prototypes-epoch{}.pth".format(60))
        # noise_save_file = str(str(config.save_dir) + "/noise-epoch{}.pth".format(60))

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes

        # prototypes = torch.load(prt_save_file).cuda()
        prototypes = self.prototypes

        noise = torch.zeros(n_new_classes, 256, device='cuda')
        noise_cnt = torch.zeros(n_new_classes, device='cuda')

        self.logger.info("computing noise...")
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)

                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)
                small_target = target[:, :, 8::16, 8::16]
                small_label = data['label'][:, 8::16, 8::16]

                normalized_features = F.normalize(features[-1], p=2, dim=1)
                class_region = small_target.unsqueeze(2) * normalized_features.unsqueeze(1)

                for cls in small_label.unique():
                    if cls in [0, 255]:
                        continue
                    dist = \
                        class_region[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:, small_label == cls] - \
                        prototypes[int(cls) - n_old_classes - 1].unsqueeze(1)
                    dist = dist ** 2

                    noise[int(cls) - n_old_classes - 1] = \
                        noise[int(cls) - n_old_classes - 1] + dist.sum(dim=1)
                    noise_cnt[int(cls) - n_old_classes - 1] = \
                        noise_cnt[int(cls) - n_old_classes - 1] + (small_label == cls).sum()

                self.progress(self.logger, batch_idx, len(self.train_loader))

            noise = torch.sqrt(noise / noise_cnt.unsqueeze(1))
            # torch.save(noise, noise_save_file)
            # self.noise = noise

            if config['data_loader']['args']['task']['step'] == 0:
                self.noise = noise
            else:
                self.noise = torch.cat([self.prev_noise, noise], dim=0)
            
        self.model.train()

    def test(self):
        result = self._test()
        
        if self.rank == 0:
            log = {}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
            test_miou = self._extract_test_miou(result)
            self._finalize_info_log(test_miou)

    def progress(self, logger, i, total_length):
        period = total_length // 5
        if period == 0:
            return
        elif (i % period == 0):
            logger.info(f'[{i}/{total_length}]')

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
            }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _save_best_model(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        """
        arch = type(self.model).__name__
        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        else:
            state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                'monitor_best': self.mnt_best,
                # 'config': self.config
            }
        best_path = str(self.checkpoint_dir / 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path, test=False):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        if not self.reset_best_mnt:
            self.mnt_best = checkpoint['monitor_best']

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'])
            # self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        if test is False:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _get_head_out_channels(self):
        model = self.model.module if isinstance(
            self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        ) else self.model
        if hasattr(model, 'tot_classes'):
            return model.tot_classes
        if hasattr(model, 'cls'):
            return sum(head.out_channels for head in model.cls)
        return None

def label_to_one_hot(label, logit, n_old_classes, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - (n_old_classes + 1)] = (label == int(cls_idx)).float()
    return target
