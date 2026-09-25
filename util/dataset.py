import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, args, config, flag='train'):
        self.args = args
        self.output_dim = config.output_dim
        self.hist_len = config.hist_len
        self.pred_len = config.pred_len
        self.train_ratio = config.train_ratio
        self.valid_ratio = config.valid_ratio
        self.pi_fen = self.args.pi_fen
        if self.train_ratio <= 0 or self.valid_ratio <= 0 or self.train_ratio + self.valid_ratio >= 1:
            raise ValueError('Invalid train/validation split ratios.')
        data = np.load(os.path.join(args.data_path, args.dataset + '.npy'))
        self.T, self.F, self.U = data.shape
        train_len = int(self.train_ratio * len(data))
        valid_len = int(self.valid_ratio * len(data))
        test_len = len(data) - train_len - valid_len
        need_len = self.hist_len + self.pred_len
        if min(train_len, valid_len, test_len) < need_len:
            raise ValueError(f'Dataset is too short for hist_len={self.hist_len} and pred_len={self.pred_len}.')
        train_end = int(self.train_ratio * self.T)
        if self.pi_fen == 1:
            valid_start = train_end
            valid_end = int((self.train_ratio + self.valid_ratio) * self.T)
            test_start = valid_end
        elif self.pi_fen == 2:
            valid_start = train_end - self.hist_len - self.pred_len + 1
            valid_end = int((self.train_ratio + self.valid_ratio) * self.T)
            test_start = valid_end - self.hist_len - self.pred_len + 1
        else:
            raise ValueError(f'Unsupported pi_fen: {self.pi_fen}')
        train_data = data[:train_end, :, :]
        self.mean = np.mean(train_data, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        self.std = np.std(train_data, axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        self.std[self.std < 1e-8] = 1.0
        is_day_idx = 1
        raw_is_day = data[:, is_day_idx:is_day_idx + 1, :].astype(np.float32).copy()
        data = data.astype(np.float32)
        data -= self.mean
        data /= self.std
        data[:, is_day_idx:is_day_idx + 1, :] = raw_is_day
        if flag == 'train':
            self.data = data[:train_end, :, :]
        elif flag == 'valid':
            self.data = data[valid_start:valid_end, :, :]
        elif flag == 'test':
            self.data = data[test_start:, :, :]
        else:
            raise ValueError(f'Invalid dataset flag: {flag}')
        self.time_samples = len(self.data) - self.hist_len - self.pred_len + 1

    def __getitem__(self, index):
        x_data = self.data[index:index + self.hist_len, :, :]
        y_data = self.data[index + self.hist_len:index + self.hist_len + self.pred_len, :, :]
        x_data = np.transpose(x_data, (2, 0, 1)).astype(np.float32)
        y_data = np.transpose(y_data, (2, 0, 1)).astype(np.float32)
        unit_idx = np.arange(self.U, dtype=np.int64)
        return x_data, y_data, unit_idx

    def inverse_transform(self, data, unit_idx):
        if isinstance(data, torch.Tensor):
            unit_idx = unit_idx.detach().cpu().numpy()
        mean = self.mean[0, :self.output_dim, unit_idx]
        std = self.std[0, :self.output_dim, unit_idx]
        if mean.ndim == 2 and mean.shape[0] == self.output_dim and mean.shape[1] == len(unit_idx):
            mean = np.transpose(mean, axes=(1, 0))
            std = np.transpose(std, axes=(1, 0))
        if data.ndim == 3:
            mean = mean[:, np.newaxis, :]
            std = std[:, np.newaxis, :]
        if isinstance(data, torch.Tensor):
            mean = torch.tensor(mean, dtype=data.dtype, device=data.device)
            std = torch.tensor(std, dtype=data.dtype, device=data.device)
        elif isinstance(data, np.ndarray):
            mean = mean.astype(np.float32)
            std = std.astype(np.float32)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}')
        return data * std + mean

    def __len__(self):
        return self.time_samples
