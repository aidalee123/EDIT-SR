"""Run EditSR on the ODE-Strogatz benchmark and print live results to stdout."""

import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from src.EditSR.architectures.model import Model
from src.EditSR.architectures.data import DataModule
from src.EditSR.dclasses import BFGSParams, FitParams, NNEquation
from src.EditSR.utils import load_metadata_hdf5,symbol_equivalence_single
from functools import partial
import hydra,math,re
from sympy import symbols, exp, lambdify,sympify
from numpy import nan
from sklearn.metrics import r2_score
import multiprocessing
import warnings
import  torch
warnings.filterwarnings("ignore")
import sympy as sp
import torch.nn.functional as F
import math
from src.EditSR.architectures.bfgs import coefficient_regularization
from src.EditSR.project_paths import scripts_path, resolve_path
# print(coefficient_regularization("(x_1/(x_1 + 1) - 0.0750000004718736*x_2)*(1.19279964553448e-6*x_1*sin(x_1)/(x_1 + 1) + 0.999999941390077*x_2)"))
from sympy import sympify, lambdify, preorder_traversal
def calculate_tree_size(expression_str):
    expr = sp.sympify(expression_str)
    # 遍历语法树的所有节点
    nodes = list(preorder_traversal(expr))
    return len(nodes)
import ast
import re


def pad_to_10_columns(tensor):
    # 检查形状是否为 [200, n]
    n = tensor.size(1)

    # 计算需要补多少列
    pad_cols = 10 - n

    # 用0补在右侧（dim=1）
    padded_tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=0)

    return padded_tensor

def get_variable_names(expr: str):
    # 找出所有变量 x_数字
    variables = re.findall(r'x_\d+', expr)

    # 去重并按编号排序
    unique_vars = sorted(set(variables), key=lambda x: int(x.split('_')[1]))

    return unique_vars

def auto_round_number(num, tolerance=1e-4, max_digits=6):
    """
    自动将 num 精简为最少小数位，确保误差在 tolerance 内。
    """
    for d in range(max_digits + 1):
        rounded = round(num, d)
        if abs(num - rounded) < tolerance:
            if rounded == int(rounded):
                return str(int(rounded))
            else:
                return str(rounded)
    return str(round(num, max_digits))
def round_if_needed(val):
    """
    对一个 sympy 的数字进行四舍五入：保留一位小数，
    如果小数部分为 0，则返回整数类型。
    """
    num = float(val)
    rounded = round(num, 1)
    if abs(rounded - int(rounded)) < 1e-10:
        return sp.Integer(int(rounded))
    else:
        return sp.Float(rounded)

def process_expr(expr, in_exponent=False):
    """
    递归遍历 sympy 表达式 expr，寻找数字常数，
    如果不在指数位置则进行四舍五入处理。

    参数：
      expr: sympy 表达式
      in_exponent: 布尔型，指示当前 expr 是否处在幂（Pow）的指数位置
    返回：
      经过处理后的表达式
    """
    # 如果 expr 是原子对象（Atom），则直接返回，
    # 因为原子对象（包括 Symbol 或其他常量）不需要遍历其子节点
    if expr.is_Atom:
        if expr.is_number and expr.free_symbols == set() and not in_exponent:
            try:
                return round_if_needed(expr)
            except Exception:
                return expr
        return expr

    # 专门处理 Pow：保持指数原样，处理底数
    if expr.func == sp.Pow:
        base = process_expr(expr.args[0])
        exponent = process_expr(expr.args[1])  # 这里将指数也进行处理
        return sp.Pow(base, exponent)

    # 对于其它表达式，递归处理其所有子节点
    new_args = tuple(process_expr(arg, in_exponent=False) for arg in expr.args)
    return expr.func(*new_args)

# 设置目录路径
directory = str(scripts_path("ode-strogatz-master", "ode-strogatz-master"))
directory2 = str(scripts_path("ode-strogatz-master", "ode.xlsx"))



from sklearn.preprocessing import MinMaxScaler
@hydra.main(config_name="config",version_base = '1.1')
def main(cfg):
    # 统一 repair 早停逻辑：使用 train-MSE 阈值，而不是 R^2
    if not hasattr(cfg.inference, 'repair_bfgs_stop_mse'):
        cfg.inference.repair_bfgs_stop_mse = 1.0e-4
    test_data = load_metadata_hdf5(str(scripts_path("data", "val_data", "10vars", "100")))

    bfgs = BFGSParams(
        activated=cfg.inference.bfgs.activated,
        n_restarts=cfg.inference.bfgs.n_restarts,
        add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
        normalization_o=cfg.inference.bfgs.normalization_o,
        idx_remove=cfg.inference.bfgs.idx_remove,
        normalization_type=cfg.inference.bfgs.normalization_type,
        stop_time=cfg.inference.bfgs.stop_time,
    )
    params_fit = FitParams(word2id=test_data.word2id,
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

    cfg.inference.use_repair = True
    cfg.inference.return_baseline_and_repair = True

    def evaluate_candidate(raw_pred_str, X_test_eval, y_test_eval, true_formula):
        pred_expr = sp.sympify(raw_pred_str)
        variables = get_variable_names(str(pred_expr))
        X_test_np = X_test_eval.detach().cpu().numpy() if isinstance(X_test_eval, torch.Tensor) else np.asarray(X_test_eval)
        y_test_np = y_test_eval.detach().cpu().numpy() if isinstance(y_test_eval, torch.Tensor) else np.asarray(y_test_eval)

        if variables:
            X_test_dict = {var: X_test_np[:, idx] for idx, var in enumerate(variables)}
            y_pre = lambdify(",".join(variables), pred_expr)(**X_test_dict)
            if isinstance(y_pre, (float, int)):
                y_pre = np.full_like(y_test_np, float(y_pre), dtype=float)
        else:
            try:
                val = float(pred_expr)
            except Exception:
                val = 0.0
            y_pre = np.full_like(y_test_np, val, dtype=float)

        if np.iscomplexobj(y_pre):
            y_pre = np.real(y_pre)
        y_pre = np.asarray(y_pre).reshape(-1)
        r2 = r2_score(y_test_np, y_pre)
        cpx = calculate_tree_size(str(sympify(str(coefficient_regularization(str(pred_expr))))))
        sr = symbol_equivalence_single(str(true_formula), str(coefficient_regularization(str(pred_expr))), variables)
        return str(pred_expr), float(r2), int(cpx), int(sr)

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    num_files = len(files)
    print(f"目录 '{directory}' 下有 {num_files} 个文件。")
    r2_i = []
    num_node_i = []
    solut_i = []
    t_i = []
    # 文件数量
    results = []

    for i in range(1):
        r2_all = []
        num_node_all = []
        solut_all = []
        t_all = []
        df = pd.read_excel(directory2)

        for filename in os.listdir(directory):
            # if filename  not in ["d_lv1.txt","d_lv2.txt","d_barmag2.txt","d_vdp1.txt"]:
            #     continue
            print("filename",filename)
            filepath = os.path.join(directory, filename)

            data = np.loadtxt(filepath,delimiter=',')
            print(f"文件名: {filename}")
            filename_no_ext = os.path.splitext(filename)[0]
            formula_str = df.loc[df['Filename'] == filename_no_ext, 'Formula'].values[0]
            print(f"formula: {formula_str}")
            print("-" * 50)  # 分隔线
            target_noise = 0.0
            iterations = 5
            n = 200
            # 初始化在不同beam_size下的结果列表
            r2_list = []  # selected candidate test R2; selection itself uses train R2 only
            train_r2_list = []
            base_r2_list = []
            repair_r2_list = []
            base_train_r2_list = []
            repair_train_r2_list = []
            solut = []
            num_node = []
            pre_expr_list = []
            # 记录开始时间
            start_time = time.time()

            # 将尝试的beam_size放在 for _ in range(iterations) 外面

            for beam in [30]:

                # 每次尝试新的beam_size，清空结果列表
                r2_list.clear()
                train_r2_list.clear()
                base_r2_list.clear()
                repair_r2_list.clear()
                base_train_r2_list.clear()
                repair_train_r2_list.clear()
                solut.clear()
                num_node.clear()
                pre_expr_list.clear()
                indices = np.random.permutation(len(data))

                # 计算训练集大小（75%）
                split_idx = int(0.75 * len(data))
                train_indices = indices[:split_idx]
                test_indices = indices[split_idx:]
                # 划分训练集和测试集
                scale = target_noise * np.sqrt(np.mean(np.square(data[train_indices, 0])))
                rng = np.random.RandomState()
                noise = rng.normal(loc=0.0, scale=scale, size=data[train_indices, 0].shape)
                data[train_indices, 0] += noise
                train_set = data[train_indices]
                test_set = data[test_indices]

                ii = 0
                for _ in range(iterations):
                    cfg.inference.beam_size =  min(50, (_ + 1) * 10)
                    ii+=1
                    sample_size = min(n, train_set.shape[0])
                    indices = np.random.choice(train_set.shape[0], size=sample_size, replace=False)
                    train_batch = train_set[indices]
                    # 构造 X 和 y：只使用当前采样 batch，避免 indices 变量被计算后未生效。
                    X = torch.tensor(train_batch[:, 1:])  # 特征列从第1列开始
                    y = torch.tensor(train_batch[:, 0].reshape(-1, 1))  # 标签是第0列
                    X_test = torch.tensor(test_set[:, 1:])
                    y_test = torch.tensor(test_set[:, 0])
                    # 每次迭代重新划分数据集

                    # === 3. 分割训练集与测试集 ===
                    # 转成 Tensor 并 pad
                    X = pad_to_10_columns(torch.tensor(X))
                    X_test = pad_to_10_columns(torch.tensor(X_test))
                    X_train_eval = pad_to_10_columns(torch.tensor(train_set[:, 1:]))
                    y_train_eval = torch.tensor(train_set[:, 0])

                    # === 创建 scaler 并拟合训练集 ===
                    try:
                        output = fitfunc(X, y.squeeze(), cfg_params=cfg.inference, test_data=test_data)
                        true_formula = sympify(formula_str)

                        base_expr_ = '-'
                        repair_expr_ = '-'
                        base_r2_ = float('nan')
                        repair_r2_ = float('nan')
                        base_train_r2_ = float('nan')
                        repair_train_r2_ = float('nan')
                        base_num_node_ = np.inf
                        repair_num_node_ = np.inf
                        base_sr_ = 0
                        repair_sr_ = 0

                        candidates = []
                        if output.get('baseline_best_bfgs_preds') is not None:
                            base_raw_pred = output['baseline_best_bfgs_preds'][0]
                            base_expr_, base_train_r2_, base_num_node_, base_sr_ = evaluate_candidate(
                                base_raw_pred, X_train_eval, y_train_eval, true_formula
                            )
                            _, base_r2_, _, _ = evaluate_candidate(
                                base_raw_pred, X_test, y_test, true_formula
                            )
                            if np.isfinite(base_train_r2_):
                                candidates.append(('base', base_train_r2_, base_r2_, base_num_node_, base_sr_, base_expr_))
                        if output.get('repair_best_bfgs_preds') is not None:
                            repair_raw_pred = output['repair_best_bfgs_preds'][0]
                            repair_expr_, repair_train_r2_, repair_num_node_, repair_sr_ = evaluate_candidate(
                                repair_raw_pred, X_train_eval, y_train_eval, true_formula
                            )
                            _, repair_r2_, _, _ = evaluate_candidate(
                                repair_raw_pred, X_test, y_test, true_formula
                            )
                            if np.isfinite(repair_train_r2_):
                                candidates.append(('repair', repair_train_r2_, repair_r2_, repair_num_node_, repair_sr_, repair_expr_))

                        print("formula:", true_formula)
                        print("base_expr:", base_expr_)
                        print("repair_expr:", repair_expr_)
                        print(f"Base Train/Test R2: {base_train_r2_:.4f}/{base_r2_:.4f}" if np.isfinite(base_train_r2_) and np.isfinite(base_r2_) else "Base Train/Test R2: nan")
                        print(f"Repair Train/Test R2: {repair_train_r2_:.4f}/{repair_r2_:.4f}" if np.isfinite(repair_train_r2_) and np.isfinite(repair_r2_) else "Repair Train/Test R2: nan")

                        if candidates:
                            _, selection_r2_, r2_, num_node_, sr, pre_expr_ = max(candidates, key=lambda item: item[1])
                        else:
                            selection_r2_ = float('-inf')
                            r2_ = 0
                            num_node_ = np.inf
                            sr = 0
                            pre_expr_ = '-'
                    except Exception as e:
                        print("异常:", e)
                        selection_r2_ = float('-inf')
                        r2_ = 0
                        base_r2_ = float('nan')
                        repair_r2_ = float('nan')
                        base_train_r2_ = float('nan')
                        repair_train_r2_ = float('nan')
                        num_node_ = np.inf
                        sr = 0
                        pre_expr_ = '-'

                    r2_list.append(r2_)
                    train_r2_list.append(selection_r2_ if np.isfinite(selection_r2_) else float('-inf'))
                    base_r2_list.append(base_r2_ if np.isfinite(base_r2_) else np.nan)
                    repair_r2_list.append(repair_r2_ if np.isfinite(repair_r2_) else np.nan)
                    base_train_r2_list.append(base_train_r2_ if np.isfinite(base_train_r2_) else np.nan)
                    repair_train_r2_list.append(repair_train_r2_ if np.isfinite(repair_train_r2_) else np.nan)
                    solut.append(sr)
                    num_node.append(num_node_)
                    pre_expr_list.append(pre_expr_)

                    print("当前迭代Selection Train R2:", np.nanmax(train_r2_list))
                    print("当前迭代Selected Test R2:", r2_)
                    print("当前sr:", sr)
                    if np.nanmax(train_r2_list) > 0.999:
                        # 满足阈值，退出当前beam_size下的迭代
                        break

                # 如果在当前beam_size下已经获得足够高的r2，则直接退出beam_size循环
                if np.nanmax(train_r2_list) >= 0.999:
                    break

            end_time = time.time()
            elapsed = end_time - start_time
            t_all.append(elapsed)

            # 只用训练集 R2 选择该文件所有迭代中的最佳候选；测试集只用于最终报告。
            best_index = int(np.nanargmax(train_r2_list))
            solut_all.append(solut[best_index])
            r2_all.append(r2_list[best_index])
            num_node_all.append(num_node[best_index])

            results.append([
                filename,
                sympify(formula_str),
                coefficient_regularization(str(pre_expr_list[best_index])),
                r2_list[best_index],
                base_r2_list[best_index],
                repair_r2_list[best_index],
                solut[best_index],
                elapsed,
                num_node[best_index],
                target_noise,
                i
            ])
            latest_row = {
                'filename': filename,
                'true_expr': sympify(formula_str),
                'pre_expr': coefficient_regularization(str(pre_expr_list[best_index])),
                'r2': r2_list[best_index],
                'selection_train_r2': train_r2_list[best_index],
                'base_r2': base_r2_list[best_index],
                'repair_r2': repair_r2_list[best_index],
                'base_train_r2': base_train_r2_list[best_index],
                'repair_train_r2': repair_train_r2_list[best_index],
                'sr': solut[best_index],
                't': elapsed,
                'Complexity': num_node[best_index],
                'noise': target_noise,
                'Iteration': i,
            }
            print(f"[Result] {latest_row['filename']} | R2={latest_row['r2']:.4f} | Base={latest_row['base_r2']:.4f} | Repair={latest_row['repair_r2']:.4f} | SR={latest_row['sr']} | Complexity={latest_row['Complexity']} | Time={latest_row['t']:.2f}s")

        r2_i.append(sum(r > 0.999 for r in r2_all) / len(r2_all))
        solut_i.append(np.mean(np.array(solut_all)))
        num_node_i.append(np.mean(np.array(num_node_all)))
        t_i.append(np.mean(np.array(t_all)))

    print("r2:", r2_i)
    print("solut:", solut_i)
    print("Complexity:", num_node_i)
    print("t:", t_i)

if __name__ == "__main__":

    main()