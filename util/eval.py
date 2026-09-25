import numpy as np
import torch


def get_model_stats(model):
    param_size = sum(param.nelement() * param.element_size() for param in model.parameters())
    buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in model.buffers())
    model_size = (param_size + buffer_size) / 1024 / 1024
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_by_module = {}
    for name, param in model.named_parameters():
        top_level_module = name.split('.')[0]
        params_by_module[top_level_module] = params_by_module.get(top_level_module, 0) + param.numel()
    return [model_size, total_params, trainable_params, params_by_module]


class Evaluate_np:
    def __init__(self, N):
        self.N = N
        self.reset()

    def reset(self):
        self.T = 0
        self.sum_abs_error = np.zeros(self.N, dtype=np.float64)
        self.sum_squared_error = np.zeros(self.N, dtype=np.float64)
        self.sum_abs_percent_error = np.zeros(self.N, dtype=np.float64)
        self.sum_x = np.zeros(self.N, dtype=np.float64)
        self.sum_y = np.zeros(self.N, dtype=np.float64)
        self.sum_xx = np.zeros(self.N, dtype=np.float64)
        self.sum_yy = np.zeros(self.N, dtype=np.float64)
        self.sum_xy = np.zeros(self.N, dtype=np.float64)

    def update(self, true, pred):
        true = true.reshape(-1, self.N)
        pred = pred.reshape(-1, self.N)
        self.T += pred.shape[0]
        abs_error = np.abs(pred - true)
        self.sum_abs_error += np.sum(abs_error, axis=0)
        self.sum_squared_error += np.sum(abs_error ** 2, axis=0)
        self.sum_abs_percent_error += np.sum(abs_error / (np.abs(true) + 1e-8), axis=0)
        self.sum_x += np.sum(true, axis=0)
        self.sum_y += np.sum(pred, axis=0)
        self.sum_xx += np.sum(true ** 2, axis=0)
        self.sum_yy += np.sum(pred ** 2, axis=0)
        self.sum_xy += np.sum(true * pred, axis=0)

    def result(self):
        if self.T == 0:
            return {'MAPE': 0.0, 'MAE': 0.0, 'MSE': 0.0, 'RMSE': 0.0, 'pearson': 0.0, 'R2': 0.0, 'R2_adj': 0.0}
        numerator = self.sum_xy - self.sum_x * self.sum_y / self.T
        var_x = self.sum_xx - self.sum_x ** 2 / self.T
        var_y = self.sum_yy - self.sum_y ** 2 / self.T
        denominator = np.sqrt(np.maximum(var_x * var_y, 0.0))
        pearson = np.zeros(self.N)
        mask = denominator > 1e-8
        pearson[mask] = numerator[mask] / denominator[mask]
        R2 = np.full(self.N, np.nan, dtype=np.float64)
        var_x_mask = var_x > 1e-8
        R2[var_x_mask] = 1.0 - self.sum_squared_error[var_x_mask] / var_x[var_x_mask]
        R2_adj = np.full(self.N, np.nan, dtype=np.float64)
        denominator_r2_adj = self.T - self.N - 1.0
        if denominator_r2_adj > 1e-8:
            factor = (self.T - 1.0) / denominator_r2_adj
            R2_adj[var_x_mask] = 1.0 - (1.0 - R2[var_x_mask]) * factor
        MAPE = self.sum_abs_percent_error / self.T * 100
        MAE = self.sum_abs_error / self.T
        MSE = self.sum_squared_error / self.T
        RMSE = np.sqrt(MSE)
        return {
            'MAPE': MAPE,
            'MAE': MAE,
            'MSE': MSE,
            'RMSE': RMSE,
            'pearson': pearson,
            'R2': R2,
            'R2_adj': R2_adj,
            'mean-MAPE': np.mean(MAPE),
            'mean-MAE': np.mean(MAE),
            'mean-MSE': np.mean(MSE),
            'mean-RMSE_feature': np.mean(RMSE),
            'mean-RMSE_all': np.sqrt(np.mean(MSE)),
            'mean-pearson': np.mean(pearson),
            'mean-R2': np.nanmean(R2),
            'mean-R2_adj': np.nanmean(R2_adj),
        }


class Evaluate_tensor:
    def __init__(self, N, device):
        self.N = N
        self.device = device
        self.reset()

    def reset(self):
        self.T = 0
        self.sum_abs_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_squared_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_abs_percent_error = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_x = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_y = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_xx = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_yy = torch.zeros(self.N, dtype=torch.float64, device=self.device)
        self.sum_xy = torch.zeros(self.N, dtype=torch.float64, device=self.device)

    def update(self, true, pred):
        if true.device != self.device or pred.device != self.device:
            raise RuntimeError(f'Input device mismatch: true={true.device}, pred={pred.device}, evaluator={self.device}')
        true = true.reshape(-1, self.N)
        pred = pred.reshape(-1, self.N)
        with torch.no_grad():
            abs_error = torch.abs(pred - true)
            self.sum_abs_percent_error += torch.sum(abs_error / (torch.abs(true) + 1e-8), dim=0)
            self.sum_abs_error += torch.sum(abs_error, dim=0)
            self.sum_squared_error += torch.sum(abs_error ** 2, dim=0)
            self.sum_x += torch.sum(true, dim=0)
            self.sum_y += torch.sum(pred, dim=0)
            self.sum_xx += torch.sum(true ** 2, dim=0)
            self.sum_yy += torch.sum(pred ** 2, dim=0)
            self.sum_xy += torch.sum(true * pred, dim=0)
            self.T += true.shape[0]

    def result(self):
        with torch.no_grad():
            MAPE = self.sum_abs_percent_error / self.T * 100
            MAE = self.sum_abs_error / self.T
            MSE = self.sum_squared_error / self.T
            RMSE = torch.sqrt(MSE)
            numerator = self.sum_xy - self.sum_x * self.sum_y / self.T
            var_x = self.sum_xx - self.sum_x ** 2 / self.T
            var_y = self.sum_yy - self.sum_y ** 2 / self.T
            denominator = torch.sqrt(torch.clamp(var_x * var_y, min=0.0))
            pearson = torch.zeros(self.N, dtype=torch.float64, device=self.device)
            mask = denominator > 1e-8
            pearson[mask] = numerator[mask] / denominator[mask]
            R2 = torch.full((self.N,), float('nan'), dtype=torch.float64, device=self.device)
            var_x_mask = var_x > 1e-8
            R2[var_x_mask] = 1.0 - self.sum_squared_error[var_x_mask] / var_x[var_x_mask]
            R2_adj = torch.full((self.N,), float('nan'), dtype=torch.float64, device=self.device)
            denominator_r2_adj = self.T - self.N - 1.0
            if denominator_r2_adj > 1e-8:
                factor = (self.T - 1.0) / denominator_r2_adj
                R2_adj[var_x_mask] = 1.0 - (1.0 - R2[var_x_mask]) * factor
            valid_r2 = ~torch.isnan(R2)
            valid_r2_adj = ~torch.isnan(R2_adj)
            mean_R2 = torch.mean(R2[valid_r2]) if valid_r2.sum() > 0 else torch.tensor(0.0, device=self.device)
            mean_R2_adj = torch.mean(R2_adj[valid_r2_adj]) if valid_r2_adj.sum() > 0 else torch.tensor(0.0, device=self.device)
            return {
                'MAPE': MAPE,
                'MAE': MAE,
                'MSE': MSE,
                'RMSE': RMSE,
                'pearson': pearson,
                'R2': R2,
                'R2_adj': R2_adj,
                'mean-MAPE': torch.mean(MAPE),
                'mean-MAE': torch.mean(MAE),
                'mean-MSE': torch.mean(MSE),
                'mean-RMSE_feature': torch.mean(RMSE),
                'mean-RMSE_all': torch.sqrt(torch.mean(MSE)),
                'mean-pearson': torch.mean(pearson),
                'mean-R2': mean_R2,
                'mean-R2_adj': mean_R2_adj,
            }
