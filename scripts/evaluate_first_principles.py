"""Run EditSR on the first-principles benchmark and print live results to stdout."""

import pandas as pd
import numpy as np
import os
import time
import re
import ast
import torch
import torch.nn.functional as F
import sympy as sp
from sympy import sympify, lambdify, preorder_traversal
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from functools import partial
import hydra
import warnings

# === 引入 EditSR 相关依赖 ===
from src.EditSR.architectures.model import Model
from src.EditSR.architectures.data import DataModule
from src.EditSR.dclasses import BFGSParams, FitParams
from src.EditSR.utils import load_metadata_hdf5, AutoMagnitudeScaler
from src.EditSR.project_paths import scripts_path, resolve_path

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 请确保此路径指向你的 datasets_first/firstprinciples 文件夹
DATASETS_ROOT = os.path.join(SCRIPT_DIR, 'datasets_first', 'firstprinciples')

target_datasets = [
    'first_principles_bode',
    'first_principles_hubble',
    'first_principles_kepler',
    'first_principles_tully_fisher',
    'first_principles_planck',
    'first_principles_ideal_gas',
    'first_principles_leavitt',
    'first_principles_newton',
    'first_principles_rydberg',
    'first_principles_schechter',
    'first_principles_absorption',
    'first_principles_supernovae_zr',
    'first_principles_supernovae_zg'
]


def load_srbench_data(dataset_name):
    # 构建可能的路径
    possible_paths = [
        os.path.join(DATASETS_ROOT, dataset_name, f"{dataset_name}.tsv.gz"),
        os.path.join(DATASETS_ROOT, dataset_name, f"{dataset_name}.tsv")
    ]

    file_path = None
    for p in possible_paths:
        if os.path.exists(p):
            file_path = p
            break

    if not file_path:
        print(f"[错误] 找不到数据文件: {dataset_name}")
        return None, None

    try:
        # [核心修复]
        # 1. header=None: 防止把第一行数字误当成表头
        # 2. compression='gzip': 自动处理压缩
        df = pd.read_csv(file_path, sep='\t', compression='gzip' if file_path.endswith('.gz') else None, header=None)

        # [核心修复] 强制转数值，处理 "x0", "x1" 等表头或 "?" 坏数据
        # errors='coerce' 会把非数字变成 NaN
        df = df.apply(pd.to_numeric, errors='coerce')

        # 删除含 NaN 的行
        df = df.dropna()

        # 强制转 float32，杜绝 object 类型
        data_values = df.values.astype(np.float32)

        if data_values.shape[0] == 0:
            print(f"[警告] {dataset_name} 数据清洗后为空！")
            return None, None

        X = data_values[:, :-1]
        y = data_values[:, -1]
        return X, y
    except Exception as e:
        print(f"[读取失败] {dataset_name}: {e}")
        return None, None


# ==============================================================================
# 3. 辅助函数
# ===========================================================================
def calculate_tree_size(expression_str):
    try:
        expr = sp.sympify(expression_str)
        nodes = list(preorder_traversal(expr))
        return len(nodes)
    except:
        return 0


def pad_to_10_columns(tensor):
    n = tensor.size(1)
    if n >= 10: return tensor[:, :10]
    pad_cols = 10 - n
    return F.pad(tensor, (0, pad_cols), mode='constant', value=0)


def get_variable_names(expr: str):
    variables = re.findall(r'x_\d+', expr)
    return sorted(set(variables), key=lambda x: int(x.split('_')[1]))




class ZScoreScaler:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, X):
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        if np.ndim(self.std) == 0:
            if self.std == 0:
                self.std = 1.0
        else:
            self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def restore_x_expression(self, expr):
        subs_dict = {}
        if np.ndim(self.mean) == 0:
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - self.mean) / self.std})
        for i in range(len(self.mean)):
            sym = sp.Symbol(f"x_{i + 1}")
            subs_dict[sym] = (sym - self.mean[i]) / self.std[i]
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):
        return expr * self.std + self.mean


class IdentityScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def restore_x_expression(self, expr):
        return expr

    def restore_y_expression(self, expr):
        return expr


class MinMaxScaler:
    def __init__(self):
        self.min = 0.0
        self.scale = 1.0

    def fit(self, X):
        X = np.asarray(X)
        self.min = np.min(X, axis=0)
        diff = np.max(X, axis=0) - self.min
        if np.ndim(diff) == 0:
            self.scale = 1.0 if diff == 0 else diff
        else:
            diff[diff == 0] = 1.0
            self.scale = diff
        return self

    def transform(self, X):
        return (X - self.min) / self.scale

    def restore_x_expression(self, expr):
        subs_dict = {}
        if np.ndim(self.min) == 0:
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - self.min) / self.scale})
        for i in range(len(self.min)):
            sym = sp.Symbol(f"x_{i + 1}")
            subs_dict[sym] = (sym - self.min[i]) / self.scale[i]
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):
        return expr * self.scale + self.min


def build_scalers(scaler_type, X_train, y_train):
    scaler_type = str(scaler_type).lower()

    if scaler_type == 'zscore':
        scaler_x = ZScoreScaler().fit(X_train)
        scaler_y = ZScoreScaler().fit(y_train)
    elif scaler_type == 'minmax':
        scaler_x = MinMaxScaler().fit(X_train)
        scaler_y = MinMaxScaler().fit(y_train)
    elif scaler_type == 'none':
        scaler_x = IdentityScaler()
        scaler_y = IdentityScaler()
    elif scaler_type == 'auto':
        scaler_x = AutoMagnitudeScaler(centering=False)
        scaler_y = AutoMagnitudeScaler(centering=False)
        scaler_x.fit(X_train, y=y_train)
        scaler_y.fit(y_train)
    else:
        raise ValueError(f"未知 SCALER_TYPE: {scaler_type}")

    return scaler_x, scaler_y


def transform_with_scalers(scaler_type, scaler_x, scaler_y, X_data, y_data):
    scaler_type = str(scaler_type).lower()
    if scaler_type == 'auto':
        X_scaled = scaler_x.transform(X_data)
        y_scaled = scaler_y.transform(y_data)
    else:
        X_scaled = scaler_x.transform(X_data)
        y_scaled = scaler_y.transform(y_data)
    return X_scaled, np.asarray(y_scaled).reshape(-1)


def restore_expression_with_scalers(expr, scaler_x, scaler_y):
    expr_step1 = scaler_x.restore_x_expression(expr)
    return scaler_y.restore_y_expression(expr_step1)


def evaluate_expression_on_data(expr, X_eval, y_eval):
    """Compute R2 for an already-restored expression on a provided split."""
    variables = get_variable_names(str(expr))

    if not variables:
        try:
            val = float(expr)
        except Exception:
            val = 0.0
        y_pred = np.full(len(y_eval), val)
    else:
        X_eval_dict = {}
        for var in variables:
            idx = int(var.split('_')[1]) - 1
            if idx < X_eval.shape[1]:
                X_eval_dict[var] = X_eval[:, idx]
            else:
                X_eval_dict[var] = np.zeros(len(y_eval))

        func = lambdify(variables, expr, modules="numpy")
        y_pred = func(**X_eval_dict)

    if isinstance(y_pred, (float, int)):
        y_pred = np.full(len(y_eval), y_pred)

    if np.iscomplexobj(y_pred):
        y_pred = y_pred.real
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

    return float(r2_score(y_eval, y_pred))

# ==============================================================================
# 4. 主程序
# ==============================================================================

@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    SCALER_TYPE = 'none'  # 可选: 'auto', 'zscore', 'minmax', 'none'

    # [请确认] 元数据路径
    metadata_path = str(scripts_path("data", "val_data", "10vars", "100"))

    if not os.path.exists(metadata_path):
        print(f"[警告] Metadata 路径不存在: {metadata_path}")

    test_data = load_metadata_hdf5(metadata_path)

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )

    params_fit = FitParams(
        word2id=test_data.word2id,
        id2word=test_data.id2word,
        una_ops=test_data.una_ops,
        bin_ops=test_data.bin_ops,
        total_variables=list(test_data.total_variables),
        total_coefficients=list(test_data.total_coefficients),
        rewrite_functions=list(test_data.rewrite_functions),
        bfgs=bfgs,
        beam_size=cfg.inference.beam_size
    )

    model_path = resolve_path(cfg.model_path, base="scripts")
    model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture)
    model.eval()
    model.to(cfg.inference.device)

    fitfunc = partial(model.fitfunc2, cfg_params=params_fit)
    results = []

    total_start = time.time()
    for _ in range(5):
        for ds_name in target_datasets:
            print(f"\n{'=' * 60}")
            print(f"正在处理数据集: {ds_name}")

            X_raw, y_raw = load_srbench_data(ds_name)
            if X_raw is None or len(X_raw) == 0:
                continue

            print(f"样本总数: {X_raw.shape[0]}, 特征数: {X_raw.shape[1]}")

            # === 设置总迭代次数 ===
            iterations = 20
            n_samples = 30

            best_train_r2_for_dataset = -np.inf
            best_record = None
            start_time_ds = time.time()

            # 1. 划分数据集
            indices = np.random.permutation(len(y_raw))
            split_point = int(len(y_raw) * 0.75)
            train_idx, test_idx = indices[:split_point], indices[split_point:]

            X_train_full = X_raw[train_idx]
            y_train_full = y_raw[train_idx]
            X_test_full = X_raw[test_idx]
            y_test_full = y_raw[test_idx]

            for iter_idx in range(iterations):
                # 动态 beam

                cfg.inference.beam_size = 50

                # 2. 采样
                n_train = X_train_full.shape[0]
                if n_train >= n_samples:
                    sub_indices = np.random.choice(n_train, size=n_samples, replace=False)
                else:
                    sub_indices = np.random.choice(n_train, size=n_samples, replace=True)

                X_curr = X_train_full[sub_indices]
                y_curr = y_train_full[sub_indices]

                # === 混合策略: 前半轮 Raw，后半轮按 SCALER_TYPE 归一化 ===
                apply_scaling = (iter_idx >= 10)

                scaler_x = IdentityScaler()
                scaler_y = IdentityScaler()
                current_scaler_type = 'none'
                scale_status_str = "[Raw   ]"

                if apply_scaling:
                    current_scaler_type = SCALER_TYPE
                    scaler_x, scaler_y = build_scalers(current_scaler_type, X_train_full, y_train_full)
                    X_input, y_input = transform_with_scalers(current_scaler_type, scaler_x, scaler_y, X_curr, y_curr)
                    scale_status_str = f"[Scaled:{current_scaler_type}]"
                else:
                    X_input, y_input = transform_with_scalers(current_scaler_type, scaler_x, scaler_y, X_curr, y_curr)

                # 转 Tensor 并 Pad (由于 load_srbench_data 修复，X_input 保证是 float32)
                X_tensor = pad_to_10_columns(torch.tensor(X_input, dtype=torch.float32))
                y_tensor = torch.tensor(y_input, dtype=torch.float32).reshape(-1, 1)

                try:
                    # 推理
                    output = fitfunc(X_tensor, y_tensor.squeeze(), cfg_params=cfg.inference, test_data=test_data)

                    raw_pred_str = output['best_bfgs_preds'][0]
                    pre_expr_sym = sp.sympify(raw_pred_str)

                    # === 还原表达式 (调用最新版 Scaler 方法) ===
                    refined_expr_sym = restore_expression_with_scalers(pre_expr_sym, scaler_x, scaler_y)

                    # 简化
                    refined_expr_str = str((str(refined_expr_sym)))
                    final_expr_sym = sp.sympify(refined_expr_str)

                    # === 评估 ===
                    train_r2 = evaluate_expression_on_data(final_expr_sym, X_train_full, y_train_full)
                    r2 = evaluate_expression_on_data(final_expr_sym, X_test_full, y_test_full)
                    complexity = calculate_tree_size(str(refined_expr_str))

                    print(
                        f"  [Iter {iter_idx:02d}] {scale_status_str} [Beam {50}] Train/Test R2: {train_r2:.4f}/{r2:.4f} | Expr: {refined_expr_str}")

                    if train_r2 > best_train_r2_for_dataset:
                        best_train_r2_for_dataset = train_r2
                        best_record = {
                            'dataset': ds_name,
                            'true_expr': "Unknown",
                            'pre_expr': refined_expr_str,
                            'r2': r2,
                            'selection_train_r2': train_r2,
                            'sr': -1,
                            'complexity': complexity,
                            'time': time.time() - start_time_ds,
                            'beam': 50,
                            'scaled': apply_scaling
                        }

                    if train_r2 > 0.99: break

                except Exception as e:
                    # print(f"  [Error] {e}")
                    continue

            if best_record:
                tag = "[Scaled]" if best_record['scaled'] else "[Raw]"
                print(f"  >>> {ds_name} 最佳 Test R2: {best_record['r2']:.4f} {tag} | Selection Train R2: {best_record['selection_train_r2']:.4f}")
                print(f"      Best expression: {best_record['pre_expr']}")
                print(f"      Complexity: {best_record['complexity']} | Elapsed: {best_record['time']:.2f}s")


        print(f"\n全部完成。总耗时: {(time.time() - total_start):.2f}s")


if __name__ == "__main__":
    main()