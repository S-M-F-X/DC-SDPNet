import argparse
import gc
import os
from time import time

import numpy as np
import torch

from solver.solver_cos import Solver
from util.config import RELEASE_VERSION, get_config
from util.seed import fixSeed


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_len', type=int, default=4, choices=[4, 8, 16])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dataset', type=str, default='pv_power_1,7_3d')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--only_test', dest='only_test', action='store_true')
    parser.add_argument('--train', dest='only_test', action='store_false')
    parser.set_defaults(only_test=True)
    parser.add_argument('--pi_fen', type=int, default=2, choices=[1, 2])
    parser.add_argument('--eval_regime', type=str, default='all', choices=['all', 'full', 'subset16', 'subset32', 'single227'])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    return parser


def resolve_project_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, path))


def append_log(log_path, message):
    print(message, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
        f.flush()


def format_result_row(name, result):
    return (
        f"{name:<12} "
        f"MAE {result['mean-MAE']:.6f} | "
        f"RMSE {result['mean-RMSE_all']:.6f} | "
        f"Pearson {result['mean-pearson']:.6f} | "
        f"R2 {result['mean-R2']:.6f} | "
        f"Time {result['test_time']:.4f}s"
    )


def build_evaluation_text(report):
    name_map = {
        'full': 'Full',
        'subset16': 'Subset16',
        'subset32': 'Subset32',
        'single227': 'Single227',
    }
    lines = [
        f"Checkpoint: {report['checkpoint']}",
        f"Model size: {report['size']:.4f} MB | Parameters: {report['total_params']} | Trainable parameters: {report['trainable_params']}",
    ]
    for key in ['full', 'subset16', 'subset32', 'single227']:
        if key in report['results']:
            lines.append(format_result_row(name_map[key], report['results'][key]))
    if len(report['results']) == 4:
        values = [report['results'][key] for key in ['full', 'subset16', 'subset32', 'single227']]
        lines.append(
            f"Avg4         MAE {np.mean([x['mean-MAE'] for x in values]):.6f} | "
            f"RMSE {np.mean([x['mean-RMSE_all'] for x in values]):.6f} | "
            f"Pearson {np.mean([x['mean-pearson'] for x in values]):.6f} | "
            f"R2 {np.mean([x['mean-R2'] for x in values]):.6f}"
        )
    lines.append(f"Total evaluation time: {report['total_time']:.4f} s")
    return '\n'.join(lines)


def main():
    args = build_parser().parse_args()
    config = get_config(args.pred_len)
    args.seed = config.seed if args.seed is None else args.seed
    args.use_gpu = torch.cuda.is_available() and not args.cpu
    args.data_path = resolve_project_path(args.data_path)

    save_name = f'DC_SDPNet_q{config.pred_len}'
    args.output_path = os.path.join(PROJECT_ROOT, 'output', save_name)
    args.model_path = os.path.join(args.output_path, 'model')
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.model_path, exist_ok=True)

    log_path = os.path.join(args.output_path, 'data.txt')
    checkpoint = os.path.join(args.model_path, 'checkpoint.pth')
    if args.only_test and not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Pretrained checkpoint not found: {checkpoint}')

    fixSeed(args.seed)

    append_log(log_path, '=' * 100)
    append_log(log_path, f'DC-SDPNet release: {RELEASE_VERSION}')
    append_log(log_path, f'Run: {save_name}')
    append_log(log_path, f'Mode: {"evaluation only" if args.only_test else "training + evaluation"}')
    append_log(log_path, f'Project root: {PROJECT_ROOT}')
    append_log(log_path, f'Dataset: {os.path.join(args.data_path, args.dataset + ".npy")}')
    append_log(log_path, f'Checkpoint: {checkpoint}')
    append_log(
        log_path,
        f'Horizon: q={config.pred_len} | d_model={config.d_model} | batch={config.batch_size} | '
        f'lr={config.lr} | weight_decay={config.weight_decay} | ramp_lambda={config.ramp_lambda}',
    )
    append_log(log_path, f'Training station scales: K={list(config.training_station_scales)}')
    append_log(log_path, f'Validation regime: Full (K=227)')
    append_log(log_path, f'Subset16 station IDs: {list(config.eval_subset16_ids)}')
    append_log(log_path, f'Subset32 station IDs: {list(config.eval_subset32_ids)}')

    solver = Solver(args, config)
    train_time = 0.0
    trained_epochs = 0
    best_epoch = 0
    best_val_loss = float('nan')

    if not args.only_test:
        append_log(log_path, 'Training started.')
        try:
            start = time()
            trained_epochs, best_epoch, best_val_loss = solver.train()
            train_time = (time() - start) / max(trained_epochs, 1)
        except KeyboardInterrupt:
            interrupt_path = os.path.join(args.output_path, 'interrupt_model')
            os.makedirs(interrupt_path, exist_ok=True)
            model_file = os.path.join(interrupt_path, 'interrupted_model.pth')
            torch.save(solver.model.state_dict(), model_file)
            append_log(log_path, f'Training interrupted. Model saved to: {model_file}')
            return
        append_log(
            log_path,
            f'Average epoch time: {train_time:.4f} s | Trained epochs: {trained_epochs} | '
            f'Best epoch: {best_epoch} | Best validation loss: {best_val_loss:.6f}',
        )

    append_log(log_path, 'Evaluation started.')
    report = solver.test_all_regimes(args.eval_regime)
    evaluation_text = build_evaluation_text(report)
    print('\nEvaluation summary:')
    print(evaluation_text)
    append_log(log_path, 'Run completed.')

    if args.use_gpu:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


if __name__ == '__main__':
    main()
