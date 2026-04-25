"""Run EditSR on selected SRBench black-box datasets and print live results to stdout."""

import pandas as pd
import numpy as np
import os
import time
import re
from sklearn.metrics import r2_score
import torch
import torch.nn.functional as F
import sympy as sp
from sympy import sympify, lambdify, preorder_traversal
from sklearn.feature_selection import SelectKBest, r_regression, f_regression
from functools import partial
import hydra
import warnings

# === 尝试导入 pmlb ===
try:
    from pmlb import fetch_data

    PMLB_AVAILABLE = True
except ImportError:
    PMLB_AVAILABLE = False
    print("[提示] 未检测到 pmlb 库，请确保本地有 CSV 文件或运行 `pip install pmlb`")

# === 引入 EditSR 相关依赖 ===
from src.EditSR.architectures.model import Model
from src.EditSR.architectures.data import DataModule
from src.EditSR.dclasses import BFGSParams, FitParams
from src.EditSR.utils import load_metadata_hdf5, AutoMagnitudeScaler
from src.EditSR.architectures.bfgs import coefficient_regularization
from src.EditSR.project_paths import scripts_path, resolve_path

warnings.filterwarnings("ignore")

# ==============================================================================
# 配置区域
# ==============================================================================

TARGET_DATASETS = [
    "1028_SWD", "1089_USCrime", "1193_BNG_lowbwt", "1199_BNG_echoMonths",
    "192_vineyard", "210_cloud", "522_pm10", "557_analcatdata_apneal",
    "579_fri_c0_250_5", "606_fri_c2_1000_10", "650_fri_c0_500_50", "678_visualizing_environmental"
]

# === 核心策略配置 ===
ITERATIONS = 15  # 总迭代次数 (Bagging 次数)
SCALE_START_ITER = 10  # 在第几次迭代后开启 Scaling
SCALER_TYPE = 'auto'  # 可选: 'auto', 'zscore', 'minmax', 'none'
SELECTED_K = 3  # 强制只选 Top-K 特征
N_SAMPLES_PER_BAG = 30  # 每次 Bagging 采样的样本数

LOCAL_DATA_DIR = str(scripts_path("srbench_blackbox_datasets"))


# ==============================================================================
# 辅助函数
# ==============================================================================

def calculate_tree_size(expression_str):
    """计算表达式复杂度 (Sympy节点数)"""
    try:
        expr = sp.sympify(expression_str)
        nodes = list(preorder_traversal(expr))
        return len(nodes)
    except:
        return 0


def get_top_k_features(X, y, k=3):
    if y.ndim == 2: y = y[:, 0]
    if X.shape[1] <= k: return list(range(X.shape[1]))
    try:
        score_func = r_regression
    except NameError:
        score_func = f_regression
    selector = SelectKBest(score_func, k=k)
    selector.fit(X, y)
    return list(np.argsort(-np.abs(selector.scores_))[:k])


def load_data(dataset_name):
    # 1. 尝试从 PMLB 加载
    if PMLB_AVAILABLE:
        try:
            df = fetch_data(dataset_name)
            return df.drop('target', axis=1).values, df['target'].values
        except:
            pass

    # 2. 尝试从本地加载
    path = os.path.join(LOCAL_DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(path):
        # 强制处理表头，防止 object 类型错误
        try:
            df = pd.read_csv(path)
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            return df.iloc[:, :-1].values, df.iloc[:, -1].values
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return None, None

    return None, None


def pad_to_10_columns(tensor):
    n = tensor.size(1)
    if n >= 10: return tensor[:, :10]
    return F.pad(tensor, (0, 10 - n), mode='constant', value=0)


def get_variable_names(expr_str):
    return sorted(set(re.findall(r'x_\d+', str(expr_str))), key=lambda x: int(x.split('_')[1]))




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

# ==============================================================================
# 主逻辑
# ==============================================================================

@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    # 统一 repair 早停逻辑：使用 train-MSE 阈值，而不是 R^2
    if not hasattr(cfg.inference, 'repair_bfgs_stop_mse'):
        cfg.inference.repair_bfgs_stop_mse = 1.0e-4
    metadata_path = str(scripts_path("data", "val_data", "10vars", "100"))
    if not os.path.exists(metadata_path):
        print(f"[错误] Metadata 不存在: {metadata_path}")
        return

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
        word2id=test_data.word2id, id2word=test_data.id2word,
        una_ops=test_data.una_ops, bin_ops=test_data.bin_ops,
        total_variables=list(test_data.total_variables),
        total_coefficients=list(test_data.total_coefficients),
        rewrite_functions=list(test_data.rewrite_functions),
        bfgs=bfgs, beam_size=cfg.inference.beam_size
    )

    model = Model.load_from_checkpoint(resolve_path(cfg.model_path, base="scripts"), cfg=cfg.architecture)
    model.eval()
    model.to(cfg.inference.device)
    fitfunc = partial(model.fitfunc2, cfg_params=params_fit)

    cfg.inference.use_repair = True
    cfg.inference.return_baseline_and_repair = True

    def evaluate_candidate(raw_pred_str, apply_scaling, scaler_x, scaler_y, X_eval, y_eval):
        pred_expr_sym = sp.sympify(raw_pred_str)
        pred_expr_sym = restore_expression_with_scalers(pred_expr_sym, scaler_x, scaler_y)

        final_expr_str = str(pred_expr_sym)
        final_expr_sym = sp.sympify(final_expr_str)
        variables = get_variable_names(final_expr_str)

        if not variables:
            try:
                val = float(final_expr_sym)
            except Exception:
                val = 0.0
            y_pred = val * np.ones(len(y_eval))
        else:
            X_eval_dict = {}
            for var in variables:
                idx = int(var.split('_')[1]) - 1
                if idx < X_eval.shape[1]:
                    X_eval_dict[var] = X_eval[:, idx]
                else:
                    X_eval_dict[var] = np.zeros(len(y_eval))
            func = lambdify(variables, final_expr_sym, modules="numpy")
            y_pred = func(**X_eval_dict)
            if isinstance(y_pred, (float, int)):
                y_pred = y_pred * np.ones(len(y_eval))

        if np.iscomplexobj(y_pred):
            y_pred = y_pred.real
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        r2 = r2_score(y_eval, y_pred)
        complexity = calculate_tree_size(str(final_expr_str))
        return final_expr_str, float(r2), int(complexity)

    all_results = []
    output_csv_path = "EditSR_bagging_results.csv"

    print(f"\n{'=' * 30} SRBench Bagging Evaluation {'=' * 30}")
    print(f"Scaler strategy for scaled rounds: {SCALER_TYPE}")
    for _ in range(1):
        # === 数据集循环 ===
        for ds_name in TARGET_DATASETS:
            print(f"\n>>> Dataset: {ds_name}")

            X_raw, y_raw = load_data(ds_name)
            if X_raw is None:
                print(f"Skipping {ds_name}: Data load failed.")
                continue

            # 1. 先划分数据集；任何监督特征选择都只能在训练集上 fit，避免测试集标签泄漏。
            indices = np.random.RandomState(42).permutation(len(y_raw))
            split_point = int(len(y_raw) * 0.75)
            train_idx, test_idx = indices[:split_point], indices[split_point:]

            X_train_raw = X_raw[train_idx]
            y_train_full = y_raw[train_idx]
            X_test_raw = X_raw[test_idx]
            y_test_full = y_raw[test_idx]

            # 2. 只在训练集上选择 Top-K 特征，再把同一列索引应用到测试集。
            top_indices = get_top_k_features(X_train_raw, y_train_full, k=SELECTED_K)
            X_train_full = X_train_raw[:, top_indices]
            X_test_full = X_test_raw[:, top_indices]

            best_train_r2 = -np.inf
            best_record = None

            # === 3. Bagging 循环 ===
            for iter_idx in range(ITERATIONS):
                apply_scaling = (iter_idx >= SCALE_START_ITER)
                current_scaler_type = SCALER_TYPE if apply_scaling else 'none'
                status_tag = f"[SCALED:{current_scaler_type}]" if apply_scaling else "[RAW   ]"

                # Bagging 采样
                n_samples = min(N_SAMPLES_PER_BAG, len(X_train_full))
                bag_indices = np.random.RandomState(iter_idx).choice(len(X_train_full), size=n_samples, replace=True)

                X_curr = X_train_full[bag_indices]
                y_curr = y_train_full[bag_indices]

                scaler_x, scaler_y = build_scalers(current_scaler_type, X_curr, y_curr)

                # 预处理 (Scale -> Pad)
                X_input, y_input = transform_with_scalers(current_scaler_type, scaler_x, scaler_y, X_curr, y_curr)

                X_tensor = pad_to_10_columns(torch.tensor(X_input, dtype=torch.float32))
                y_tensor = torch.tensor(y_input, dtype=torch.float32).reshape(-1, 1)

                try:
                    # 动态 Beam Size
                    cfg.inference.beam_size = 50
                    output = fitfunc(X_tensor, y_tensor.squeeze(), cfg_params=cfg.inference, test_data=test_data)

                    base_r2 = float('nan')
                    repair_r2 = float('nan')
                    base_train_r2 = float('nan')
                    repair_train_r2 = float('nan')
                    base_expr_str = None
                    repair_expr_str = None
                    base_complexity = -1
                    repair_complexity = -1

                    base_preds = output.get('baseline_best_bfgs_preds')
                    repair_preds = output.get('repair_best_bfgs_preds')

                    candidates = []
                    if base_preds is not None:
                        base_expr_str, base_train_r2, base_complexity = evaluate_candidate(
                            base_preds[0], apply_scaling, scaler_x, scaler_y, X_train_full, y_train_full
                        )
                        _, base_r2, _ = evaluate_candidate(
                            base_preds[0], apply_scaling, scaler_x, scaler_y, X_test_full, y_test_full
                        )
                        if np.isfinite(base_train_r2):
                            candidates.append(('base', base_train_r2, base_r2, base_expr_str, base_complexity))

                    if repair_preds is not None:
                        repair_expr_str, repair_train_r2, repair_complexity = evaluate_candidate(
                            repair_preds[0], apply_scaling, scaler_x, scaler_y, X_train_full, y_train_full
                        )
                        _, repair_r2, _ = evaluate_candidate(
                            repair_preds[0], apply_scaling, scaler_x, scaler_y, X_test_full, y_test_full
                        )
                        if np.isfinite(repair_train_r2):
                            candidates.append(('repair', repair_train_r2, repair_r2, repair_expr_str, repair_complexity))

                    if candidates:
                        _, selection_train_r2, r2, final_expr_str, final_complexity = max(candidates, key=lambda item: item[1])
                    else:
                        selection_train_r2 = float('-inf')
                        r2 = float('nan')
                        final_expr_str = None
                        final_complexity = -1

                    base_r2_str = f"{base_train_r2:.4f}/{base_r2:.4f}" if np.isfinite(base_train_r2) and np.isfinite(base_r2) else "nan"
                    repair_r2_str = f"{repair_train_r2:.4f}/{repair_r2:.4f}" if np.isfinite(repair_train_r2) and np.isfinite(repair_r2) else "nan"
                    print(f"  Iter {iter_idx:02d} {status_tag} Base Train/Test R2: {base_r2_str} | Repair Train/Test R2: {repair_r2_str}")

                    # 更新最佳结果：只用训练集 R2 选候选；测试集 R2 只随最终候选记录。
                    if np.isfinite(selection_train_r2) and selection_train_r2 > best_train_r2:
                        best_train_r2 = selection_train_r2
                        best_record = {
                            'dataset': ds_name,
                            'r2': r2,
                            'selection_train_r2': selection_train_r2,
                            'expression': final_expr_str,
                            'scaled': apply_scaling,
                            'iter': iter_idx,
                            'complexity': final_complexity,
                            'features': str(top_indices),
                            'base_r2': base_r2,
                            'repair_r2': repair_r2,
                            'base_train_r2': base_train_r2,
                            'repair_train_r2': repair_train_r2,
                            'base_expr': base_expr_str,
                            'repair_expr': repair_expr_str,
                        }
                        print(f"  Iter {iter_idx:02d} {status_tag} *New Best* Train R2: {selection_train_r2:.4f} | Test R2: {r2:.4f}")

                    if best_train_r2 > 0.999: break

                except Exception as e:
                    # print(f"  Iter {iter_idx:02d} Error: {e}")
                    pass

            if best_record:
                tag = "[Scaled]" if best_record['scaled'] else "[Raw]"
                print(f"  >>> {ds_name} 最佳 Test R2: {best_record['r2']:.4f} {tag} | Selection Train R2: {best_record['selection_train_r2']:.4f}")
                print(f"      Best expression: {best_record['expression']}")
                print(f"      Complexity: {best_record['complexity']} | Features: {best_record['features']}")
        print("\nAll datasets processing complete.")

if __name__ == "__main__":
    main()