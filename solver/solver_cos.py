import gc
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models.DC_SDPNet import Model as DC_SDPNet
from util.dataset import MyDataset
from util.eval import Evaluate_np, Evaluate_tensor, get_model_stats


class Solver:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.out_dim = self.config.output_dim
        self.log_path = os.path.join(self.args.output_path, 'data.txt')
        self.device = self._acquire_device()
        self.train_loader, self.valid_loader, self.test_loader = self._get_loader()
        self.channel = self.train_loader.dataset.data.shape[1]
        self.num_units = self.train_loader.dataset.data.shape[2]
        self.model = DC_SDPNet(self.config, self.channel, self.num_units).to(self.device)
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.scaler = None
        if not self.args.only_test:
            self.scaler = GradScaler('cuda', enabled=self.args.use_gpu)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=self.config.eta_min,
            )
            self.Evaluate_train = Evaluate_tensor(self.out_dim, self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device(f'cuda:{self.args.device}')
            self._log(f'Device: cuda:{self.args.device}')
            return device
        self._log('Device: cpu')
        return torch.device('cpu')

    def _log(self, message):
        print(message, flush=True)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

    def _get_loader(self):
        train_set = MyDataset(self.args, self.config, flag='train')
        valid_set = MyDataset(self.args, self.config, flag='valid')
        test_set = MyDataset(self.args, self.config, flag='test')
        self._log(f'Train shape: {train_set.data.shape}')
        self._log(f'Validation shape: {valid_set.data.shape}')
        self._log(f'Test shape: {test_set.data.shape}')
        train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_set, batch_size=self.config.batch_size, shuffle=False, drop_last=False)
        return train_loader, valid_loader, test_loader

    def _validate_eval_ids(self, ids):
        ids = [int(x) for x in ids]
        if len(ids) == 0:
            raise ValueError('Evaluation station list is empty.')
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate station IDs detected: {ids}')
        for unit_id in ids:
            if unit_id < 0 or unit_id >= self.num_units:
                raise ValueError(f'Unit ID {unit_id} is outside the valid range [0, {self.num_units - 1}].')
        return ids

    def _select_train_positions(self, U, device):
        mode = self.config.unit_train_mode
        if mode == 'full':
            return torch.arange(U, dtype=torch.long, device=device)
        if mode == 'single':
            return torch.randperm(U, device=device)[:1]
        if mode == 'subset':
            K = int(np.random.choice(self.config.unit_subset_sizes))
            K = max(1, min(K, U))
            return torch.randperm(U, device=device)[:K]
        if mode == 'mixed':
            r = np.random.rand()
            if r < self.config.unit_single_prob:
                K = 1
            elif r < self.config.unit_single_prob + self.config.unit_subset_prob:
                K = int(np.random.choice(self.config.unit_subset_sizes))
                K = max(1, min(K, U))
            else:
                K = U
            if K == U:
                return torch.arange(U, dtype=torch.long, device=device)
            return torch.randperm(U, device=device)[:K]
        raise ValueError(f'Unknown unit_train_mode: {mode}')

    def _process_one_batch(self, x_data, y_data, unit_idx, phase, eval_ids=None):
        B, U, _, _ = x_data.shape
        q = y_data.shape[2]
        x_data = x_data.float().to(self.device)
        y_data = y_data.float().to(self.device)
        unit_idx = unit_idx.long().to(self.device)
        if phase == 'train':
            select_pos = self._select_train_positions(U, self.device)
        elif phase == 'valid':
            if self.config.validation_mode != 'full':
                raise ValueError(f'Unsupported validation_mode: {self.config.validation_mode}')
            select_pos = torch.arange(U, dtype=torch.long, device=self.device)
        elif phase == 'test':
            if eval_ids is None:
                select_pos = torch.arange(U, dtype=torch.long, device=self.device)
            else:
                select_pos = torch.tensor(self._validate_eval_ids(eval_ids), dtype=torch.long, device=self.device)
        else:
            raise ValueError(f'Unknown phase: {phase}')
        x_data = x_data.index_select(1, select_pos)
        y_data = y_data.index_select(1, select_pos)
        unit_idx = unit_idx.index_select(1, select_pos)
        B, K, _, _ = x_data.shape
        future_is_day = y_data[..., 1]
        pred = self.model(x_data, unit_idx)
        true = y_data[..., :self.out_dim]
        pred = pred[..., :self.out_dim]
        true_flat = true.reshape(B * K, q, self.out_dim)
        pred_flat = pred.reshape(B * K, q, self.out_dim)
        unit_flat = unit_idx.reshape(B * K)
        true_flat = self.train_loader.dataset.inverse_transform(true_flat, unit_flat)
        pred_flat = self.train_loader.dataset.inverse_transform(pred_flat, unit_flat)
        day_flat = future_is_day.reshape(B * K, q, 1).float()
        expanded_mask = F.max_pool1d(
            day_flat.transpose(1, 2),
            kernel_size=9,
            stride=1,
            padding=4,
        ).transpose(1, 2)
        pred_flat = pred_flat * expanded_mask
        return true_flat, pred_flat

    def _training_loss(self, pred, true):
        pred = pred.float()
        true = true.float()
        base_loss = F.smooth_l1_loss(
            pred,
            true,
            beta=self.config.smooth_l1_beta,
            reduction='mean',
        )
        if pred.shape[1] > 1:
            delta_pred = pred[:, 1:, :] - pred[:, :-1, :]
            delta_true = true[:, 1:, :] - true[:, :-1, :]
            ramp_loss = F.smooth_l1_loss(
                delta_pred,
                delta_true,
                beta=self.config.smooth_l1_beta,
                reduction='mean',
            )
        else:
            ramp_loss = pred.new_zeros(())
        return base_loss + self.config.ramp_lambda * ramp_loss

    def _process_one_epoch(self, data_loader, phase):
        objective_sum = 0.0
        objective_weight = 0
        for x_data, y_data, unit_idx in data_loader:
            if phase == 'train':
                self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type, enabled=self.args.use_gpu):
                true, pred = self._process_one_batch(x_data, y_data, unit_idx, phase)
                loss = self._training_loss(pred, true)
            batch_weight = pred.shape[0]
            objective_sum += loss.detach().float().item() * batch_weight
            objective_weight += batch_weight
            self.Evaluate_train.update(true.detach().float(), pred.detach().float())
            if phase == 'train':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        metrics = self.Evaluate_train.result()
        self.Evaluate_train.reset()
        metrics['objective'] = objective_sum / max(objective_weight, 1)
        return metrics

    def _update_checkpoint(self, val_loss, epoch):
        if val_loss < self.best_val_loss:
            previous = self.best_val_loss
            self.best_val_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.no_improve = 0
            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, 'checkpoint.pth'))
            self._log(f'Checkpoint updated: {previous:.8f} -> {self.best_val_loss:.8f} at epoch {self.best_epoch}')
            return False
        self.no_improve += 1
        self._log(f'No validation improvement: {self.no_improve}/{self.config.patience}')
        return self.no_improve >= self.config.patience

    def train(self):
        trained_epochs = 0
        for epoch in range(1, self.config.epoch + 1):
            trained_epochs = epoch
            start = time()
            self.model.train()
            train_res = self._process_one_epoch(self.train_loader, 'train')
            with torch.no_grad():
                self.model.eval()
                valid_res = self._process_one_epoch(self.valid_loader, 'valid')
            elapsed = time() - start
            current_lr = self.optimizer.param_groups[0]['lr']
            self._log(
                f"Epoch {epoch}/{self.config.epoch} | {elapsed:.2f}s | "
                f"LR {current_lr:.8f} | "
                f"Train loss {train_res['objective']:.6f} | "
                f"Train MAE {train_res['mean-MAE']:.6f} | "
                f"Validation loss {valid_res['objective']:.6f} | "
                f"Validation MAE {valid_res['mean-MAE']:.6f}"
            )
            stop_training = self._update_checkpoint(valid_res['objective'], epoch)
            self.scheduler.step()
            if stop_training:
                self._log(f'Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}.')
                break
        return trained_epochs, self.best_epoch, self.best_val_loss

    def _load_checkpoint(self):
        checkpoint = os.path.join(self.args.model_path, 'checkpoint.pth')
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
        state_dict = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        return checkpoint

    def _evaluate_ids(self, eval_ids=None):
        evaluator = Evaluate_np(self.out_dim)
        start = time()
        with torch.inference_mode():
            for x_data, y_data, unit_idx in self.test_loader:
                with autocast(device_type=self.device.type, enabled=self.args.use_gpu):
                    true, pred = self._process_one_batch(x_data, y_data, unit_idx, 'test', eval_ids)
                evaluator.update(
                    true.detach().float().cpu().numpy(),
                    pred.detach().float().cpu().numpy(),
                )
        return evaluator.result(), time() - start

    def _evaluate_single227(self):
        station_results = []
        station_times = []
        for station_id in range(self.num_units):
            result, elapsed = self._evaluate_ids([station_id])
            station_results.append(result)
            station_times.append(elapsed)
        keys = ['mean-MAE', 'mean-MSE', 'mean-RMSE_all', 'mean-pearson', 'mean-R2', 'mean-R2_adj']
        averaged = {key: float(np.mean([result[key] for result in station_results])) for key in keys}
        averaged['station_count'] = self.num_units
        averaged['station_results'] = station_results
        return averaged, float(np.sum(station_times))

    def _format_eval_result(self, name, result):
        return (
            f"{name:<12} "
            f"MAE {result['mean-MAE']:.6f} | "
            f"RMSE {result['mean-RMSE_all']:.6f} | "
            f"Pearson {result['mean-pearson']:.6f} | "
            f"R2 {result['mean-R2']:.6f} | "
            f"Time {result['test_time']:.4f}s"
        )

    def test_all_regimes(self, eval_regime='all'):
        if self.args.use_gpu:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        checkpoint = self._load_checkpoint()
        model_stats = get_model_stats(self.model)
        self._log(f'Loaded checkpoint: {checkpoint}')
        self._log(
            f'Model size: {model_stats[0]:.4f} MB | Parameters: {model_stats[1]} | '
            f'Trainable parameters: {model_stats[2]}'
        )
        requested = ['full', 'subset16', 'subset32', 'single227'] if eval_regime == 'all' else [eval_regime]
        display_names = {
            'full': 'Full',
            'subset16': 'Subset16',
            'subset32': 'Subset32',
            'single227': 'Single227',
        }
        results = {}
        total_time = 0.0
        for regime in requested:
            self._log(f'Evaluating {display_names[regime]}...')
            if regime == 'full':
                result, elapsed = self._evaluate_ids(None)
            elif regime == 'subset16':
                result, elapsed = self._evaluate_ids(self.config.eval_subset16_ids)
            elif regime == 'subset32':
                result, elapsed = self._evaluate_ids(self.config.eval_subset32_ids)
            elif regime == 'single227':
                result, elapsed = self._evaluate_single227()
            else:
                raise ValueError(f'Unknown evaluation regime: {regime}')
            result['test_time'] = elapsed
            results[regime] = result
            total_time += elapsed
            self._log(self._format_eval_result(display_names[regime], result))

        if len(results) == 4:
            ordered = [results[key] for key in ['full', 'subset16', 'subset32', 'single227']]
            self._log(
                f"Avg4         MAE {np.mean([x['mean-MAE'] for x in ordered]):.6f} | "
                f"RMSE {np.mean([x['mean-RMSE_all'] for x in ordered]):.6f} | "
                f"Pearson {np.mean([x['mean-pearson'] for x in ordered]):.6f} | "
                f"R2 {np.mean([x['mean-R2'] for x in ordered]):.6f}"
            )
        self._log(f'Total evaluation time: {total_time:.4f} s')

        return {
            'checkpoint': checkpoint,
            'results': results,
            'total_time': total_time,
            'size': model_stats[0],
            'total_params': model_stats[1],
            'trainable_params': model_stats[2],
            'params_by_module': model_stats[3],
        }
