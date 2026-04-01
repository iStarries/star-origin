import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from models.loss import WBCELoss, PKDLoss, ContLoss,calculate_certainty
from utils.gmm import GaussianMixture


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, config, logger, gpu):
        self.config = config
        
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['epochs'] if cfg_trainer['save_period'] == -1 else cfg_trainer['save_period']

        self.validation_period = cfg_trainer['validation_period'] if cfg_trainer['validation_period'] == -1 else cfg_trainer['validation_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        self.reset_best_mnt = cfg_trainer['reset_best_mnt']
        self.rank = torch.distributed.get_rank()

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
        validate_on_test = self.config.config.get('validate', False)
        validate_on_val = self.config.config.get('validate_a', False)
        self.tail_validation_sources = []
        if validate_on_val:
            self.tail_validation_sources.append('val')
        if validate_on_test:
            self.tail_validation_sources.append('test')
        self.enable_tail_validation = len(self.tail_validation_sources) > 0
        self.tail_validation_window = 25
        self.tail_best_miou = {'val': -inf, 'test': -inf}
        self.tail_best_path = {'val': None, 'test': None}
        self.log_file_path = self.config.log_dir / "info.log"


        # if config.resume is not None:
        #     self._resume_checkpoint(config.resume)

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

        if self.config['data_loader']['args']['task']['step'] ==0:
            self.epochs = self.config['trainer']['epochs']
        else:
            self.epochs = self.config['trainer']['epochs_incre']
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, val_flag = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            if self.enable_tail_validation and epoch >= self.epochs - self.tail_validation_window + 1:
                for source in self.tail_validation_sources:
                    if source == 'test':
                        tail_log = self._test(epoch)
                        prefix = 'test_'
                    elif source == 'val':
                        tail_log = self._valid_epoch(epoch)
                        prefix = 'val_'
                    else:
                        raise RuntimeError(f"Unsupported tail validation source: {source}")

                    log.update(**{prefix + k: v for k, v in tail_log.items()})
                    if self.rank == 0:
                        tail_miou = self._extract_tail_miou(tail_log)
                        if tail_miou is not None and tail_miou > self.tail_best_miou[source]:
                            self.tail_best_miou[source] = tail_miou
                            self._save_best_tail_checkpoint(epoch, tail_miou, source)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            if self.rank == 0:

                if val_flag and (self.mnt_mode != 'off'):
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and float(log[self.mnt_metric]) <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and float(log[self.mnt_metric])-self.mnt_best >= 0)

                    except KeyError:
                        self.logger.warning("Warning: Metric '{}' is not found. "
                                            "Model performance monitoring is disabled.".format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False
                    if self.config['data_loader']['args']['task']['step'] == 0:
                        improved = True
                    if improved:
                        self.mnt_best = float(log[self.mnt_metric])
                        not_improved_count = 0

                    else:
                        not_improved_count += 1

                if (self.early_stop > 0) and (not_improved_count >= self.early_stop):
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(epoch))
                    self.writer.close()
                    break

                if epoch % self.save_period == 0:

                    if self.config['data_loader']['args']['task']['step'] > 0 and epoch % int(self.epochs/2) == 0:
                        self.refine_gmms()

                    self._save_checkpoint(self.epochs)
                    self.compute_gmms(self.config)
                    self.save_gmms(self.config, self.epochs)

        if self.rank == 0:
            self._log_tail_best_results()
        # close TensorboardX
        self.writer.close()

    def refine_gmms(self):
        self.logger.info("refine gmms for old classes...")

        pred_numbers = torch.zeros(self.n_old_classes + 1).to(self.device)
        fes_old = {k: [] for k in range(self.n_old_classes)}
        fes_gen = {k: [] for k in range(self.n_old_classes)}
        for batch_idx, data in enumerate(self.train_loader):
            with torch.no_grad():
                data['image'], data['label'] = data['image'].to(self.device), data['label'].to(self.device)
                logit_old, features_old, _ = self.model_old(data['image'], ret_intermediate=True)
                logit_new, features_new, _ = self.model(data['image'], ret_intermediate=True)

                logit_old = logit_old.detach()
                pred_old = logit_old.argmax(dim=1) + 1  # pred: [N. H, W]
                idx_old = (logit_old > 0.5).float()  # logit: [N, C, H, W]
                idx_old = idx_old.sum(dim=1)  # logit: [N, H, W]
                pred_old[idx_old == 0] = 0  # set background (non-target class)


                pred_region_old = (pred_old * (data['label'] == 0))
                target_old = label_to_one_hot_old_cls(pred_region_old, logit_old)

                small_pred_region_old = pred_region_old[:, 8::32, 8::32]
                small_target_old = target_old[:, :, 8::32, 8::32]

                for i in range(1, self.n_old_classes + 1):
                    pred_numbers[i] = pred_numbers[i] + (small_pred_region_old == i).sum()

                features = F.interpolate(
                    features_new[-1], size=(16, 16),
                    mode="bilinear", align_corners=False
                )

                old_class_region = small_target_old.unsqueeze(2) * features.unsqueeze(1)


                for cls in small_pred_region_old.unique():
                    if cls in [0,255]:
                        continue
                    fes_old[int(cls) - 1].append(old_class_region[:, int(cls)- 1].permute(1, 0, 2, 3)[:,
                        (small_pred_region_old == cls)])
                self.progress(self.logger, batch_idx, len(self.train_loader))
        for k in range(self.n_old_classes):
            if len(fes_old[k]) <= 1:
                continue
            fes_old[k] = torch.cat(fes_old[k], dim=1).to(self.device)
            gen_numbers = int(self.prev_numbers[k+1]/2)
            fes_gen[k] = self.prev_fes_model[k].cpu().sample(int(gen_numbers))[0].to(self.device)
            fes = torch.cat([fes_old[k], fes_gen[k].permute(1, 0)], dim=1).to(self.device)


            self.prev_fes_model[k].to(self.device).fit(fes.permute(1, 0))

    def save_gmms(self, config, epoch):
        save_file = str(config.save_dir) + "/prototypes-epoch{}.pth".format(epoch)
        if config['data_loader']['args']['task']['step'] == 0:
            all_info = {
                "numbers": self.numbers,
                "fes_model":self.fes_model
            }
        else:
            all_info = {
                "numbers": self.numbers,
                "fes_model": self.fes_model
            }
        torch.save(all_info, save_file)

    def compute_cls_number(self, config):
        self.logger.info("computing number of pixels...")

        number_save_file = str(config.save_dir) + "/numbers_tmp.pth"
        if os.path.exists(number_save_file):
            self.numbers = torch.load(number_save_file,map_location='cpu')
            return

        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes
        numbers = torch.zeros(n_new_classes + 1).to(self.device)

        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                small_label = data['label'][:, 8::16, 8::16].to(self.device)
                for i in range(n_new_classes + 1):
                    if i == 0:
                        numbers[i] = numbers[i] + torch.sum(small_label == 0).item()
                        continue
                    numbers[i] = numbers[i] + torch.sum(small_label == i + n_old_classes).item()
                self.progress(self.logger, batch_idx, len(self.train_loader))
        self.numbers = numbers

        torch.save(numbers, number_save_file)



    def compute_gmms(self, config):
        n_new_classes = self.n_new_classes
        n_old_classes = self.n_old_classes
        fes =  {k: [] for k in range(n_new_classes)}
        fes_model = {k+n_old_classes: GaussianMixture(config['hyperparameter']['gaus'], 256).to(self.device) for k in range(n_new_classes)}
        self.logger.info("computing gmms for every new class...")
        with torch.no_grad():
            self.model.eval()
            for batch_idx, data in enumerate(self.train_loader):

                logit, features, _ = self.model(data['image'].cuda(), ret_intermediate=True)
                pred = torch.argmax(logit,dim=1) + 1  # pred: [N. H, W]
                idx = (logit > 0.5).float()  # logit: [N, C, H, W]
                idx = idx.sum(dim=1)  # logit: [N, H, W]
                pred[idx == 0] = 0
                target = label_to_one_hot(data['label'], logit[:, -n_new_classes:], n_old_classes)

                small_target1 = target[:, :, 8::32, 8::32]
                small_label1 = data['label'][:, 8::32, 8::32]


                features[-1] = F.interpolate(
                    features[-1], size=(16,16),
                    mode="bilinear", align_corners=False
                )


                class_region1 = small_target1.unsqueeze(2) * features[-1].unsqueeze(1)


                for cls in small_label1.unique():
                    if cls in [0,255]:
                        continue
                    fes[int(cls) - n_old_classes - 1].append(class_region1[:, int(cls) - n_old_classes - 1].permute(1, 0, 2, 3)[:,
                        (small_label1 == cls)])
                self.progress(self.logger, batch_idx, len(self.train_loader))

            for k in range(n_new_classes):
                fes[k]=torch.cat(fes[k],dim=1)
                fes_model[k+n_old_classes].fit(fes[k].permute(1,0))


            if config['data_loader']['args']['task']['step'] == 0:
                self.fes_model = fes_model
            else:
                self.fes_model = {**self.prev_fes_model, **fes_model}

    def _extract_tail_miou(self, log_data):
        if not log_data:
            return None
        miou_key = 'Mean_Intersection_over_Union_overall'
        if miou_key not in log_data:
            self.logger.warning(f"Warning: Metric '{miou_key}' is not found in log.")
            return None
        return float(log_data[miou_key])

    def _save_best_tail_checkpoint(self, epoch, miou, source):
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

        filename = self.checkpoint_dir / f'{source}_best-epoch{epoch}-miou{miou:.2f}.pth'
        prev_best_path = self.tail_best_path.get(source)
        if prev_best_path is not None and prev_best_path.exists():
            prev_best_path.unlink()
        torch.save(state, filename)
        self.tail_best_path[source] = filename
        self.logger.info(f"Saving current best {source} checkpoint: {filename} ...")

    def _log_tail_best_results(self):
        for source in self.tail_validation_sources:
            best_miou = self.tail_best_miou.get(source, -inf)
            best_path = self.tail_best_path.get(source)
            if best_miou != -inf:
                self.logger.info(f"[tail-{source}] best mIoU: {best_miou:.2f}, path: {best_path}")
            else:
                self.logger.info(f"[tail-{source}] no best checkpoint recorded")

    def _finalize_info_log(self, miou_from_test=None):
        if self.rank != 0:
            return

        self._log_tail_best_results()

        best_test_miou = self.tail_best_miou.get('test', -inf)
        if best_test_miou != -inf:
            miou = best_test_miou
        else:
            miou = miou_from_test

        suffix = f"{float(miou):.2f}" if miou is not None and miou != -inf else "unknown"
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

    def _extract_test_miou(self, result):
        if result is None:
            return None
        key = "Mean_Intersection_over_Union_overall"
        if key in result:
            return result[key]
        for result_key, value in result.items():
            if result_key.endswith(key):
                return value
        return None

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

        if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model.load_state_dict(checkpoint['state_dict'])


        
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

def label_to_one_hot(label, logit, n_old_classes, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - (n_old_classes + 1)] = ((label == int(cls_idx))).float()
    return target




def label_to_one_hot_old_cls(label, logit, ignore_index=255):
    target = torch.zeros_like(logit, device='cuda').float()
    for cls_idx in label.unique():
        if cls_idx in [0, ignore_index]:
            continue
        target[:, int(cls_idx) - 1] = ((label == int(cls_idx))).float()
    return target
