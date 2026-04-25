from src.EditSR.utils import coefficient_regularization
import numpy as np
from scipy.optimize import minimize
import sympy as sp
from ..dataset.generator import Generator
from . import data
import time
from ..dataset.sympy_utils import add_multiplicative_constants, add_additive_constants
import torch
import re
from sklearn.metrics import r2_score


# ... [保留原有的 helper 函数 replace_illegal_variables 和 TimedFun 不变] ...
def replace_illegal_variables(expr, max_var=5):
    present_vars = set(re.findall(r'x_\d+', expr))
    if 'x_0' in present_vars:
        raise ValueError(f"表达式中包含非法变量 x_0：\n{expr}")
    expr_fixed = expr
    for i in range(2, max_var + 1):
        xi = f'x_{i}'
        xi_prev = f'x_{i - 1}'
        if xi in present_vars and xi_prev not in present_vars:
            expr_fixed = re.sub(rf'\bx_{i}\b', xi_prev, expr_fixed)
    return expr_fixed


class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


modules = {"numpy": np, "log": np.log, "ln": np.log, "exp": np.exp, "sin": np.sin, "cos": np.cos,
           "Abs": np.abs, "tan": np.tan, "sqrt": np.sqrt, "E": np.e, "pi": np.pi, "e": np.e,
           'asin': np.arcsin, "atan": np.arctan}


def bfgs(pred_str, X, y, cfg, test_data):
    # --- [原有逻辑开始] ---
    y = y.squeeze()
    X = X.clone()

    if isinstance(pred_str, list):
        pred_str = np.array(pred_str)

    pred_str = pred_str[1:].tolist()
    raw = data.de_tokenize(pred_str, test_data.id2word)

    if cfg.bfgs.add_coefficients_if_not_existing and 'constant' not in raw:
        variables = {x: sp.Symbol(x, real=True, nonzero=True) for x in test_data.total_variables}
        infix = Generator.prefix_to_infix(raw, coefficients=test_data.total_coefficients,
                                          variables=test_data.total_variables)
        s = Generator.infix_to_sympy(infix, variables, test_data.rewrite_functions)
        placeholder = {x: sp.Symbol(x, real=True, nonzero=True) for x in ["cm", "ca"]}
        s = add_multiplicative_constants(s, placeholder["cm"], unary_operators=test_data.una_ops)
        s = add_additive_constants(s, placeholder, unary_operators=test_data.una_ops)
        s = s.subs(placeholder["cm"], 0.43)
        s = s.subs(placeholder["ca"], 0.421)
        s_simplified = data.constants_to_placeholder(s, symbol="constant")
        prefix = Generator.sympy_to_prefix(s_simplified)
        candidate = Generator.prefix_to_infix(prefix, coefficients=["constant"], variables=test_data.total_variables)
    else:
        candidate = Generator.prefix_to_infix(raw, coefficients=["constant"], variables=test_data.total_variables)


    # Ensure variable-index consistency with the dataset (e.g., x_1..x_10).
    # `replace_illegal_variables` enforces the constraint: if x_n exists then x_{n-1} must exist.
    try:
        _vars = getattr(test_data, "total_variables", None)
        if _vars:
            max_var = max(int(str(v).split("_")[1]) for v in _vars if str(v).startswith("x_"))
        else:
            max_var = 10
    except Exception:
        max_var = 10
    candidate = replace_illegal_variables(candidate.format(constant="constant"), max_var=int(max_var))
    expr = candidate
    for i in range(candidate.count("constant")):
        expr = expr.replace("constant", f"c{i}", 1)

    if cfg.bfgs.idx_remove:
        bool_con = (X < 200).all(axis=2).squeeze()
        X = X[:, bool_con, :]

    # 这里的 diffs 实际上是符号表达式列表
    diffs = []
    for i in range(X.shape[1]):
        curr_expr = expr
        for idx, j in enumerate(test_data.total_variables):
            curr_expr = sp.sympify(curr_expr).subs(j, X[:, i, idx])
        diff = curr_expr - y[i]
        diffs.append(diff)

    # 构建符号损失函数 'loss'
    if cfg.bfgs.normalization_type == "NMSE":
        if isinstance(y, torch.Tensor):
            mean_y = np.mean(y.cpu().numpy())
        else:
            mean_y = np.mean(y)
        loss = (np.mean(np.square(diffs))) / mean_y if abs(mean_y) > 1e-06 else (np.mean(np.square(diffs)))
    elif cfg.bfgs.normalization_type == "MSE":
        loss = (np.mean(np.square(diffs)))
    else:
        raise KeyError

    F_loss = []
    consts_ = []
    funcs = []
    symbols = {i: sp.Symbol(f'c{i}') for i in range(candidate.count("constant"))}
    all_symbols_list = list(symbols.values())  # [c0, c1, c2...]

    # 第一轮：全局优化
    for i in range(cfg.bfgs.n_restarts):
        x0 = np.random.randn(len(symbols)) * 10
        fun_timed = TimedFun(fun=sp.lambdify(all_symbols_list, loss, modules=modules), stop_after=cfg.bfgs.stop_time)

        def safe_loss(c):
            try:
                val = fun_timed.fun(c)
                if np.isnan(val) or np.isinf(val): return 1e6
                return val
            except Exception:
                return 1e6

        if len(x0):
            res = minimize(safe_loss, x0, method='BFGS')
            consts_.append(fun_timed.x)  # 使用最后一次计算的x，或者 res.x
        else:
            consts_.append([])

        # 构建当前轮次的表达式
        final = expr
        current_consts = consts_[-1]
        for idx_c, val_c in enumerate(current_consts):
            final = sp.sympify(final).replace(symbols[idx_c], val_c)
        funcs.append(final)

        # 计算 Loss (验证)
        values = {x: X[:, :, idx].cpu() for idx, x in enumerate(test_data.total_variables)}
        try:
            y_found = sp.lambdify(",".join(test_data.total_variables), final, modules=modules)(**values)
            final_loss = np.mean(np.square(y_found - y.cpu()).numpy())
        except:
            final_loss = 1e9
        F_loss.append(final_loss)

    try:
        k_best = np.nanargmin(F_loss)
    except ValueError:
        k_best = 0

    best_expr_str = str(funcs[k_best])
    best_consts = consts_[k_best]
    best_loss = F_loss[k_best]

    # --- [修改版功能：智能稀疏化 (保护单常数乘法项)] ---

    # 阈值设定
    PRUNE_THRESHOLD = float(cfg.bfgs.prune_threshold)
    LOSS_TOLERANCE = float(cfg.bfgs.prune_tolerance)

    if len(best_consts) > 0:
        # 1. 初步筛选：找出绝对值小于阈值的常数索引
        potential_zero_indices = [idx for idx, val in enumerate(best_consts) if abs(val) < PRUNE_THRESHOLD]

        final_to_zero_indices = []

        # 2. 智能过滤：检查是否可以安全置0
        for idx in potential_zero_indices:
            c_sym = symbols[idx]

            # 策略：如果只有一个常数，我们需要非常谨慎
            if len(best_consts) == 1:
                # 计算表达式对该常数的导数
                # expr 是包含 c0, c1... 的符号模板
                derivative = sp.diff(expr, c_sym)

                # 核心判断：
                # 如果导数包含符号变量 (如 x_1)，说明是 c*x (乘性) -> 导数为 x -> is_constant()为False -> 跳过(保护)
                # 如果导数只是数字 (如 1)，说明是 x+c (加性) -> 导数为 1 -> is_constant()为True -> 允许置0
                if not derivative.is_constant():
                    # print(f"检测到常数 {c_sym} 为乘性项 (0.001*x 形式)，跳过稀疏化")
                    continue

            final_to_zero_indices.append(idx)

        # 3. 执行稀疏化重优化逻辑
        if len(final_to_zero_indices) > 0:
            # print(f"尝试置零索引: {final_to_zero_indices}")

            subs_map = {symbols[idx]: 0.0 for idx in final_to_zero_indices}
            remaining_indices = [idx for idx in range(len(best_consts)) if idx not in final_to_zero_indices]
            remaining_symbols = [symbols[idx] for idx in remaining_indices]

            # 符号层面上置0
            pruned_sym_loss = loss.subs(subs_map)

            # 准备重优化的初值
            x0_pruned = np.array([best_consts[i] for i in remaining_indices])

            if len(x0_pruned) > 0:
                pruned_func_lambdified = sp.lambdify(remaining_symbols, pruned_sym_loss, modules=modules)
                fun_timed_pruned = TimedFun(fun=pruned_func_lambdified, stop_after=cfg.bfgs.stop_time)

                def safe_loss_pruned(c):
                    try:
                        val = fun_timed_pruned.fun(c)
                        if np.isnan(val) or np.isinf(val): return 1e6
                        return val
                    except Exception:
                        return 1e6

                res_pruned = minimize(safe_loss_pruned, x0_pruned, method='BFGS')
                optimized_remaining_consts = res_pruned.x

                # 重组参数 map
                final_consts_map = {}
                for idx in final_to_zero_indices:
                    final_consts_map[symbols[idx]] = 0.0
                for i, r_idx in enumerate(remaining_indices):
                    final_consts_map[symbols[r_idx]] = optimized_remaining_consts[i]
            else:
                final_consts_map = {symbols[idx]: 0.0 for idx in range(len(best_consts))}

            # 生成最终表达式
            final_pruned_expr = expr
            for s_sym, val in final_consts_map.items():
                final_pruned_expr = sp.sympify(final_pruned_expr).replace(s_sym, val)

            # 验证 Loss
            try:
                values = {x: X[:, :, idx].cpu() for idx, x in enumerate(test_data.total_variables)}
                y_found_pruned = sp.lambdify(",".join(test_data.total_variables), final_pruned_expr, modules=modules)(
                    **values)
                pruned_loss = np.mean(np.square(y_found_pruned - y.cpu()).numpy())
            except:
                pruned_loss = 1e9

            # 比较策略
            is_acceptable = False
            if best_loss == 0:
                if pruned_loss < 1e-9: is_acceptable = True
            elif pruned_loss <= best_loss * LOSS_TOLERANCE:
                is_acceptable = True

            if is_acceptable:
                best_expr_str = str(final_pruned_expr)
                best_loss = pruned_loss
                best_consts = [final_consts_map[symbols[i]] for i in range(len(best_consts))]

    return best_expr_str, best_consts, best_loss, expr