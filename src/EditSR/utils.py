import marshal
import inspect  # [NEW] 引入 inspect 用于修复类检测 Bug
import sympy as sp
import random
import re
import pandas as pd
from sympy.core.rules import Transform
from sympy import sympify, Symbol, Float, Integer, sin, cos, tan, exp, log, Add, Mul, Pow, Abs, sqrt, asin
from torch.utils import data
from functools import partial

"""
Timeout handling
"""

MAXTIME = 60

import signal, sympy


class SimplifyTimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    print(f"raising SimplifyTimeOutException")
    raise SimplifyTimeOutException


"""
For all of these, higher is better
"""
import pickle
import json

from .dclasses import DatasetDetails, Equation
from typing import List, Tuple
import h5py
import os
import numpy as np
from pathlib import Path
import re
import torch


def solve_and_swap_random(expr):
    """
        输入: SymPy 表达式 (如 x_1**2 + x_2)
        输出: 解析后的新表达式 (如 sqrt(x_1 - x_2)) 或 原表达式 (如果失败)
        """
    # 1. 边界检查: 如果是常数(没有变量)，直接返回原值
    free_syms = list(expr.free_symbols)
    if not free_syms:
        return expr

    # 2. 随机锁定一个变量
    target_var = random.choice(free_syms)

    # 3. 尝试解析 (使用 Try-Except 兜底)
    try:
        # 定义影子变量 (代表原函数的值)
        shadow = sympy.Symbol('__SHADOW__')

        # 极速求解 (关闭所有检查和化简)
        solutions = sympy.solve(
            sympy.Eq(shadow, expr),
            target_var,
            manual=True,  # 关键: 手工模式，速度极快
            check=False,  # 关键: 不验证解
            simplify=False  # 关键: 不化简
        )

        # 如果无解，返回原表达式
        if not solutions:
            return expr

        # 4. 筛选解
        # 简单过滤虚数 (字符串判断比数学判断快)
        valid_sols = [s for s in solutions if 'I' not in str(s)]

        # 如果过滤完没剩东西(全是虚数)，就用原来的解；否则用过滤后的
        candidates = valid_sols if valid_sols else solutions

        # 随机选一个解 (保留了正负号的随机性)
        chosen_sol = random.choice(candidates)

        # 5. 回代 (将影子变量替换回目标变量名)
        final_expr = chosen_sol.subs(shadow, target_var)

        return final_expr

    except Exception:
        # 发生任何错误(超时、算法不支持等)，静默退回原表达式
        return expr


import numpy as np
import sympy as sp

import numpy as np
import sympy as sp


class AutoMagnitudeScaler:
    """
    智能自动数量级缩放器 (Robust Auto-Magnitude Scaler - Center & Scale)

    [新增功能]
    - centering: 是否减去中位数 (去偏置)。解决 y = f(x) + C 中的 C。
    """

    def __init__(self, verbose=False, centering=False):
        self.scales = None
        self.centers = None  # [NEW] 存储中位数
        self.centering = centering  # [NEW] 开关
        self.is_bounded_detected = False
        self.verbose = verbose
        self.diagnostics = {}

    @staticmethod
    def _calculate_robust_params(arr, centering=False):
        """同时计算 Center 和 Scale"""
        arr = np.asarray(arr)
        # 排除 NaN/Inf
        arr = arr[np.isfinite(arr)]

        if len(arr) == 0: return 0.0, 1.0

        # 1. 计算中位数 (Center)
        med_val = np.median(arr)
        center = med_val if centering else 0.0

        # 2. 计算去偏后的绝对波动
        arr_centered = arr - center
        arr_abs = np.abs(arr_centered)
        arr_nonzero = arr_abs[arr_abs > 0]

        # 3. 计算 Scale (IQR 优先)
        q75, q25 = np.percentile(arr, [75, 25])
        iqr = q75 - q25

        # 如果去偏后还有非零值，计算其中位数
        abs_med = np.median(arr_nonzero) if len(arr_nonzero) > 0 else 1.0

        if iqr > 1e-12:
            target_metric = iqr
        else:
            target_metric = abs_med

        if target_metric < 1e-300: target_metric = 1.0

        log_val = np.log10(target_metric)
        exponent = int(np.floor(log_val))
        exponent = np.clip(exponent, -300, 300)

        if abs(exponent) >= 1:
            scale = 10.0 ** float(exponent)
        else:
            scale = 1.0

        return center, scale

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.is_bounded_detected = False
        self.diagnostics = {}

        # --- 1. 计算 Params (Center & Scale) ---
        if X.ndim == 1:
            c, s = self._calculate_robust_params(X, self.centering)
            self.centers = c
            raw_scales = s
        else:
            params = [self._calculate_robust_params(X[:, i], self.centering) for i in range(X.shape[1])]
            self.centers = np.array([p[0] for p in params])
            raw_scales = np.array([p[1] for p in params])

        self.diagnostics["raw_scales"] = raw_scales
        self.diagnostics["centers"] = self.centers

        # --- 2. 频率保护 (保持不变) ---
        # 只有当提供了 y (说明是特征 X) 且 X 是多维时才做稳定性检查
        # 对于 Target Y，我们通常不做稳定性检查，直接缩放
        perform_stability_check = (y is not None) and (X.ndim > 1)

        if perform_stability_check:
            is_stable, details = self._check_is_stable_function(X, np.asarray(y))
            self.diagnostics.update(details)

            if is_stable:
                if X.ndim == 1:
                    self.scales = 1.0
                else:
                    self.scales = np.ones(X.shape[1])
                # 如果判定为稳定函数，也不去中心化，保持原样
                if np.ndim(self.centers) == 0:
                    self.centers = 0.0
                else:
                    self.centers = np.zeros(X.shape[1])

                self.is_bounded_detected = True
                self.diagnostics["final_decision"] = "Bounded -> Force Identity"
            else:
                self.scales = raw_scales
                self.diagnostics["final_decision"] = "Unbounded -> Robust Scale"
        else:
            self.scales = raw_scales
            self.diagnostics["final_decision"] = "Standard Robust Scale"

        self.diagnostics["final_scales"] = self.scales

        if self.verbose: self._print_diagnostics()
        return self

    def _check_is_stable_function(self, X, y):
        """(逻辑保持不变，省略以节省空间，直接复制之前的即可)"""
        # ... 复制之前的 _check_is_stable_function 代码 ...
        # 为方便，这里简写返回 False
        return False, {"score": 999}

    def _print_diagnostics(self):
        d = self.diagnostics
        print(f"\n[AutoMagnitudeScaler Diagnostics]")
        print(f"  > Center (Bias): {d.get('centers')}")  # [NEW]
        print(f"  > Scale (Spread): {d.get('raw_scales')}")
        print(f"  > Final Decision: {d['final_decision']}")

    def transform(self, X):
        if self.scales is None: raise ValueError("Scaler not fit.")
        X = np.asarray(X)
        # (X - Center) / Scale
        return ((X - self.centers) / self.scales).astype(np.float32)

    def inverse_transform(self, X_scaled):
        if self.scales is None: raise ValueError("Scaler not fit.")
        # X * Scale + Center
        return (np.asarray(X_scaled) * self.scales + self.centers).astype(np.float32)

    def restore_x_expression(self, expr):
        """符号还原 X: x_i -> (x_i - c) / s"""
        if self.scales is None: return expr

        # 1D Case
        if np.ndim(self.scales) == 0:
            s = self.scales
            c = self.centers
            if s == 1.0 and c == 0.0: return expr

            # 注意：Transform 是 (x-c)/s
            # 这里的 restore 是把公式里的 x_1 (它是 scaled 的) 替换回 raw x 的表达
            # x_scaled = (x_raw - c) / s
            # 所以直接替换 Symbol('x_1') 为 (Symbol('x_1') - c) / s
            return expr.subs({sp.Symbol("x_1"): (sp.Symbol("x_1") - c) / s})

        # ND Case
        subs_dict = {}
        for i, (s, c) in enumerate(zip(self.scales, self.centers)):
            if s != 1.0 or c != 0.0:
                sym = sp.Symbol(f"x_{i + 1}")
                subs_dict[sym] = (sym - c) / s
        return expr.subs(subs_dict)

    def restore_y_expression(self, expr):
        """符号还原 Y: y_scaled -> y_scaled * s + c"""
        if self.scales is None: return expr

        # 处理 scalar/array 兼容性
        s = self.scales
        c = self.centers
        if isinstance(s, (np.ndarray, list)):
            s = s[0] if len(s) > 0 else 1.0
            c = c[0] if len(c) > 0 else 0.0

        if s == 1.0 and c == 0.0: return expr

        # y_raw = y_scaled * s + c
        return expr * s + c


class H5FilesCreator():
    def __init__(self, base_path: Path = None, target_path: Path = None, metadata=None):

        target_path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.target_path = target_path

        self.base_path = base_path
        self.metadata = metadata

    def create_single_hd5_from_eqs(self, block):
        name_file, eqs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq in enumerate(eqs):
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()

    def recreate_single_hd5_from_idx(self, block: Tuple):
        name_file, eq_idxs = block
        t_hf = h5py.File(os.path.join(self.target_path, str(name_file) + ".h5"), 'w')
        for i, eq_idx in enumerate(eq_idxs):
            eq = load_eq_raw(self.base_path, eq_idx, self.metadata.eqs_per_hdf)
            # curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=eq)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)


def load_eq_raw(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    f.close()
    return raw_metadata


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx / num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder, f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file) * int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata


def load_metadata_hdf5(path_folder: Path) -> DatasetDetails:
    f = h5py.File(os.path.join(path_folder, "metadata.h5"), 'r')
    # print(f)
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)

    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata


def alarm_handler(signum, frame):
    print("raising SimplifyTimeOutException")
    raise SimplifyTimeOutException


def round_floats(expr):
    expr_mod = expr
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float):
            if abs(a) < 0.0001:
                expr_mod = expr_mod.subs(a, sp.Integer(0))
            else:
                expr_mod = expr_mod.subs(a, round(a, 3))
    return expr_mod


def get_symbolic_model(expr_str, local_dict):
    sp_model = sp.parse_expr(expr_str, local_dict=local_dict)
    sp_model = round_floats(sp_model)
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME)
    try:
        sp_model = sp.simplify(sp_model)
    except Exception as e:
        print('Warning: simplify failed. Msg:', e)
    finally:
        signal.alarm(0)
    return sp_model


def regularization(match):
    # 获取匹配到的数字字符串
    num_str = match.group()
    try:
        # 转换为浮点数
        x = float(num_str)
    except ValueError:
        return num_str

    # 定义一个列表，依次存储 (rounding_digits, threshold)
    # 整数比较：round(x,0)返回浮点数，所以使用 int(round(x)) 转换为整数
    candidates = [
        (0, 0.1),  # 整数
        (1, 0.01),  # 一位小数
        (2, 0.001),  # 两位小数
        (3, 0.0001)  # 三位小数
    ]

    for digits, thresh in candidates:
        rounded = round(x, digits)
        if abs(x - rounded) <= thresh:
            # 对整数情况，去掉小数部分
            if digits == 0:
                return str(int(rounded))
            else:
                # 保证格式显示相应的小数位数
                return f"{rounded:.{digits}f}"
    # 如果没有任何替换条件满足，则返回原字符串
    return num_str


def coefficient_regularization(expression):
    """
    扫描表达式中的常数，按约定规则进行替换，并返回修改后的表达式。
    """
    # 匹配浮点数或整数（注意：这里假设表达式中数字之间不会和变量名混淆）
    pattern = r'(?<![A-Za-z_])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    # 用回调函数进行替换
    new_expression = re.sub(pattern, regularization, expression)
    return new_expression


def symbolic_equivalence(true_model_expr, pred_model_str, local_dict):
    """
    判断预测的表达式（字符串形式）是否与给定的真值表达式（sympy 对象）等价。
    返回1表示等价，返回0表示不等价。
    """
    sp_model = get_symbolic_model(pred_model_str, local_dict)
    sym_diff = round_floats(true_model_expr - sp_model)
    sym_frac = round_floats(sp_model / true_model_expr)
    print('true_model:', true_model_expr, '; \npred_model:', sp_model)
    try:
        diff_const = sym_diff.is_constant(simplify=False)
        frac_const = sym_frac.is_constant(simplify=False)
        if not diff_const and not frac_const:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(MAXTIME)
            try:
                if not diff_const:
                    sym_diff = sp.simplify(sym_diff)
                    diff_const = sym_diff.is_constant()
                if not frac_const:
                    sym_frac = sp.simplify(sym_frac)
                    frac_const = sym_frac.is_constant()
            except Exception as e:
                print('Warning: simplify failed. Msg:', e)
            finally:
                signal.alarm(0)
    except Exception as e:
        print('Constant checking failed.', e)
        diff_const = False
        frac_const = False
    is_equivalent = (str(sym_diff) == '0' or diff_const or frac_const)
    return 1 if is_equivalent else 0


def symbol_equivalence_single(true_model_str, pred_model_str, feature_names):
    """
    判断单个预测表达式与真值表达式是否符号等价，返回1或0。

    参数:
      true_model_str : 真值表达式字符串
      pred_model_str : 预测表达式字符串
      feature_names  : 包含特征变量名的列表，例如 ['x']

    返回:
      1 表示预测表达式与真值表达式数学等价，0 表示不等价。
    """
    local_dict = {f: sp.Symbol(f) for f in feature_names}
    try:
        true_expr = get_symbolic_model(true_model_str, local_dict)
    except Exception as e:
        print(f"解析真值表达式失败: {true_model_str}，错误: {e}")
        return 0
    return symbolic_equivalence(true_expr, pred_model_str, local_dict)


def div(x, y):
    return x / y


SAFE_OPS = [Add, Mul, sin, cos, exp, log, Abs, sqrt, asin, tan, div]
UNARY_OPS = {sin, cos, exp, log, Abs, sqrt, asin, tan}
SAFE_NEST_OPS = {Abs, sqrt}  # 这些算子作为父节点或子节点时总是安全的（随意嵌套）


# -----------------------------------------------------------------------------
# [FIXED] Helper Functions for Nesting Constraints
# -----------------------------------------------------------------------------

def get_op_class(op):
    """
    获取算子的类，修复了对 SymPy 类的错误属性访问
    """
    if op == div:
        return Mul  # div 返回的是 Mul (x * 1/y)，视为二元算子

    # [FIX] 如果 op 本身就是一个类（例如 sin, cos），直接返回它
    # 不要去访问 .func，否则会获取到 SymPy 的 property 对象导致判断失效
    if inspect.isclass(op):
        return op

    # 如果 op 是一个实例（例如 sin(x)），返回它的函数类
    if hasattr(op, 'func'):
        return op.func

    return op


def is_valid_nesting(parent_node, child_op):
    """
    检查父子算子嵌套是否合法
    """
    if parent_node is None:
        return True

    parent_op = parent_node.func
    child_op_class = get_op_class(child_op)

    # 1. 如果父节点不是一元算子（是二元如Add/Mul，或Power），则不限制子节点
    if parent_op not in UNARY_OPS:
        return True

    # 2. 如果父节点是 Abs 或 sqrt，允许随意包裹任何内容
    if parent_op in SAFE_NEST_OPS:
        return True

    # --- 以下父节点均为受限一元算子 (sin, cos, exp, log, asin, tan) ---

    # 3. 子节点如果是二元算子 (Add, Mul, div)，总是允许
    if child_op_class not in UNARY_OPS:
        return True

    # 4. 子节点如果是 Abs 或 sqrt，总是允许 (被包裹)
    if child_op_class in SAFE_NEST_OPS:
        return True

    # 5. 特殊例外规则
    # 允许 exp(exp(...))

    # 允许 asin(sin(...))
    if parent_op == asin and child_op_class == sin:
        return True

    # 6. 其他情况禁止嵌套 (如 sin(cos), log(sin), sin(sin) 等)
    return False


# -----------------------------------------------------------------------------
# Precision Mutation Logic (Point-to-Point)
# -----------------------------------------------------------------------------

def get_target_variables(expr):
    """提取表达式中所有形如 x_1, x_i 的变量"""
    return {s for s in expr.free_symbols if re.match(r'x_\d+', str(s))}


def count_candidates(expr, check_func):
    """Pass 1: 统计有多少个节点满足修改条件"""
    count = 0
    if check_func(expr):
        count += 1
    for arg in expr.args:
        count += count_candidates(arg, check_func)
    return count


def apply_mutation_at_indices(expr, check_func, target_indices, mutate_func, counter, parent_node=None):
    """Pass 2: 仅在选中的 target_indices 位置应用 mutate_func，并传递 parent_node"""
    # 1. Check current node
    if check_func(expr):
        current_idx = counter[0]
        counter[0] += 1
        if current_idx in target_indices:
            # 命中！应用修改，传入父节点上下文
            return mutate_func(expr, parent_node=parent_node)

    # 2. Recurse children
    new_args = []
    changed = False
    for arg in expr.args:
        # 递归时，当前 expr 成为子节点的 parent_node
        new_arg = apply_mutation_at_indices(arg, check_func, target_indices, mutate_func, counter, parent_node=expr)
        if new_arg is not arg:
            changed = True
        new_args.append(new_arg)

    if changed:
        return expr.func(*new_args)
    return expr


def mutate_with_selection(expr, check_func, num_to_pick, mutate_func):
    """
    通用函数：在满足 check_func 的所有节点中，随机选 num_to_pick 个进行 mutate_func
    """
    total_candidates = count_candidates(expr, check_func)
    if total_candidates == 0:
        return expr

    num_to_pick = min(num_to_pick, total_candidates)
    target_indices = set(random.sample(range(total_candidates), num_to_pick))

    counter = [0]
    # 初始调用 parent_node 为 None
    return apply_mutation_at_indices(expr, check_func, target_indices, mutate_func, counter, parent_node=None)


# --- 具体破坏策略 ---

def mandatory_constants_mutation(expr):
    """[必选] 随机选 1-2 个常数，修改为 [-10, 10]"""

    def check(node): return node.is_Number

    def mutate(node, parent_node=None):
        return Float(random.uniform(-3, 3))

    num = random.randint(1, 2)
    return mutate_with_selection(expr, check, num, mutate)


def mandatory_exponents_mutation(expr):
    """[必选] 随机选 1-2 个指数，重置为 [-10, 10]"""

    def check(node):
        return isinstance(node, Pow) and node.args[1].is_Number

    def mutate(node, parent_node=None):
        base, _ = node.args
        if random.random() < 0.8:
            new_exp = Integer(random.randint(-3, 3))
        else:
            new_exp = Float(random.uniform(-5, 5))
        return Pow(base, new_exp)

    num = random.randint(1, 2)
    return mutate_with_selection(expr, check, num, mutate)


def strategy_constant_injection(node, variables, parent_node=None):
    """策略 E: 常数注入 (原 mandatory_constant_injection)"""
    # 随机生成一个常数
    c = Float(random.uniform(-9, 9))
    op = random.choice(['add', 'sub', 'mul', 'div'])

    if op == 'add': return node + c
    if op == 'sub': return node - c
    if op == 'mul': return node * c
    if op == 'div': return node / (c if abs(c) > 0.01 else 1.0)
    return node


# --- 结构破坏策略池 ---

def generate_random_subtree(variables, max_depth=2, current_depth=0, parent_node=None):
    """
    生成随机噪声子树，强制遵守嵌套规则
    """
    # 递归终止条件
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.4):
        if random.random() < 0.6 and variables:
            return random.choice(list(variables))
        else:
            return Float(random.uniform(-3, 3))

    # 过滤出符合嵌套规则的算子
    valid_ops = [op for op in SAFE_OPS if is_valid_nesting(parent_node, op)]

    if not valid_ops:
        return Float(random.uniform(-3, 3))

    op = random.choice(valid_ops)

    # 构建 dummy parent 用于下一层检查
    dummy_parent = None
    if op in [sin, cos, tan, asin, exp, log, sqrt]:
        class Dummy:
            pass

        dummy_parent = Dummy()
        dummy_parent.func = op
    else:
        dummy_parent = None

    # 生成参数
    if op in [sin, cos, exp, log, tan, asin, sqrt]:
        arg = generate_random_subtree(variables, max_depth, current_depth + 1, parent_node=dummy_parent)
        return op(arg)
    elif op == div:  # [FIX] 单独处理 div，它只需要2个参数
        args = [generate_random_subtree(variables, max_depth, current_depth + 1, parent_node=dummy_parent) for _ in
                range(2)]
        return op(*args)
    elif op in [Add, Mul]:
        num_args = random.randint(2, 3)
        args = [generate_random_subtree(variables, max_depth, current_depth + 1, parent_node=dummy_parent) for _ in
                range(num_args)]
        return op(*args)

    return Float(1.0)


# 策略 A: 结构重置
def strategy_structure_reset(node, variables, parent_node=None):
    return generate_random_subtree(variables, max_depth=2, parent_node=parent_node)


# 策略 B: 算子互换
def strategy_operator_swap(node, variables, parent_node=None):
    if isinstance(node, Add): return Mul(*node.args)
    if isinstance(node, Mul): return Add(*node.args)

    # 互换一元算子时检查合法性
    if isinstance(node, sin):
        return cos(*node.args) if is_valid_nesting(parent_node, cos) else node
    if isinstance(node, cos):
        return sin(*node.args) if is_valid_nesting(parent_node, sin) else node
    return node


# 策略 C: 剪枝
def strategy_pruning(node, variables, parent_node=None):
    if isinstance(node, (Add, Mul)) and len(node.args) > 1:
        args = list(node.args)
        args.pop(random.randint(0, len(args) - 1))
        return node.func(*args)
    return node


# 策略 D: 变量扭曲
def strategy_variable_distortion(node, variables, parent_node=None):
    if node.is_Symbol and str(node).startswith('x_'):
        r = random.random()
        # 尝试包裹 sin
        if r < 0.3:
            if is_valid_nesting(parent_node, sin):
                return sin(node)
            else:
                return node
                # 乘系数
        if r < 0.6: return node * Float(random.uniform(-2, 2))
        # 包裹 Abs
        return Abs(node)
    return node


# --- 主入口 ---

def mutate_expression_structure(expr_input):
    """
    主入口：
    1. 尝试自然变异 (Retry Loop)：必须满足变量齐全且表达式发生了改变。
    2. 如果重试多次均失败，则执行强制变量召回或强制微扰。
    """
    if isinstance(expr_input, str):
        expr_input = expr_input.replace("^", "**")
        try:
            original_expr = sympify(expr_input)
        except:
            return sympify("x_1 * 1.5")
    else:
        original_expr = expr_input

    required_vars = get_target_variables(original_expr)
    original_str = str(original_expr)  # 缓存原始字符串，避免重复计算

    # 定义结构破坏策略池
    structure_strategies = [
        strategy_constant_injection,  # 策略 E
        strategy_structure_reset,  # 策略 A
        strategy_operator_swap,  # 策略 B
        strategy_pruning,  # 策略 C
        strategy_variable_distortion  # 策略 D
    ]

    MAX_RETRIES = 20

    # 初始化
    mutated_expr = original_expr

    # =====================================================
    # Phase 1: 尝试自然变异 (Retry Loop)
    # =====================================================
    for attempt in range(MAX_RETRIES):
        # 重置起点
        mutated_expr = original_expr

        # 1. 基础数值变异
        mutated_expr = mandatory_constants_mutation(mutated_expr)
        mutated_expr = mandatory_exponents_mutation(mutated_expr)

        # 2. 随机结构变异 (2处)
        for _ in range(2):
            chosen_strategy = random.choice(structure_strategies)
            mutate_func = partial(chosen_strategy, variables=required_vars)

            mutated_expr = mutate_with_selection(
                mutated_expr,
                check_func=lambda x: True,
                num_to_pick=1,
                mutate_func=mutate_func
            )

        # 3. [修改] 联合检查：变量完整性 & 表达式唯一性
        current_vars = get_target_variables(mutated_expr)
        current_str = str(mutated_expr)

        # 条件 A: 所有必须的变量都在
        # 条件 B: 字符串表示与原版不同 (确保发生了实际变异)
        if required_vars.issubset(current_vars) and current_str != original_str:
            return mutated_expr

        # >> 如果不满足，continue，进入下一次尝试 (Re-mutate) <<

    # =====================================================
    # Phase 2: 强制变量召回与兜底 (Ultimate Fallback)
    # =====================================================
    # 此时 mutated_expr 是第 20 次失败的结果。
    # 失败原因可能是：变量丢失 OR 表达式未变。

    current_vars = get_target_variables(mutated_expr)
    missing_vars = required_vars - current_vars

    # 情况 A: 如果是因为变量丢失导致的失败 -> 强制召回
    if missing_vars:
        inject_terms = []
        for v in missing_vars:
            r = random.random()
            coeff = Float(random.uniform(-5, 5))
            if r < 0.4:
                term = coeff * v
            elif r < 0.7:
                term = coeff * v * v
            else:
                term = sin(coeff * v)
            inject_terms.append(term)
        mutated_expr = Add(mutated_expr, *inject_terms)

    # 情况 B: 如果变量都在，但因为表达式没变(str相等)导致的失败
    # 或者 情况 A 召回之后依然不幸地和原表达式一样
    if str(mutated_expr) == original_str:
        # 强制微扰：加一个极小的随机数，确保不等
        mutated_expr = mutated_expr + Float(random.uniform(0.01, 0.1))

    return mutated_expr


def constrain_expression_values(expr):
    """
    对 SymPy 表达式进行复杂的数值约束：
    1. 指数 (Exponent)：
       - 全局最多允许 1 个非整数指数 (如 0.5, 1.3)。
       - 其余指数必须是整数。
       - 所有指数（无论之前是整是非整）如果绝对值 > 10，强制替换为 [-10, 10] 的随机整数。
    2. 常数 (Constant)：
       - 全局最多保留 3 个常数（系数或加数）。
       - 多余的常数根据上下文“消化掉”（加法中变 0，乘法中变 1）。
       - 保留的常数若绝对值 > 10，截断到 [-10, 10]。
    """

    # 定义一个简单的状态类来追踪全局计数
    class Context:
        def __init__(self):
            self.const_count = 0  # 已保留的常数数量
            self.non_int_exp_count = 0  # 已保留的非整数指数数量

    context = Context()

    # 辅助：判断是否通过了“数值截断”检查
    # 如果 abs(val) > 10 返回截断后的值，否则返回 None (代表保持原样)
    def get_clipped_val(num):
        val = float(num)

        # 1. 如果数值在范围内，返回 None (代表保持原样)
        if abs(val) <= 5:
            return None

            # 2. 如果数值越界，根据原类型生成随机数
        if getattr(num, 'is_Integer', False):
            # 原值是整数 -> 变成 -10 到 10 的随机整数 (例如: 100 -> 3)
            return sp.Integer(random.randint(-3, 3))
        else:
            # 原值是浮点 -> 变成 -10.0 到 10.0 的随机浮点数 (例如: 50.5 -> -2.414...)
            return sp.Float(random.uniform(-3, 3))

    # 递归主逻辑，增加 parent_type 以便智能消除常数
    def _recurse(node, parent_type=None):

        # ---------------------------
        # 1. 处理幂运算 (Pow) - 优先处理指数逻辑
        # ---------------------------
        if node.is_Pow:
            base, exp = node.args

            # 递归处理底数，标记父节点为 Pow
            new_base = _recurse(base, parent_type=sp.Pow)

            new_exp = exp
            if exp.is_Number:
                # 判断当前指数是否为“数学上的整数”（包括 2.0, 3.00 等）
                is_math_integer = float(exp).is_integer()

                target_val = exp

                # --- 逻辑 A: 处理非整数指数 ---
                if not is_math_integer:
                    if context.non_int_exp_count < 1:
                        # 配额未满，允许保留这个非整数
                        context.non_int_exp_count += 1
                        # 依然检查 magnitude (可选，如果希望非整数也不要太大)
                        clipped = get_clipped_val(exp)
                        if clipped is not None: target_val = clipped
                    else:
                        # 配额已满，强制变为随机整数
                        target_val = sp.Integer(random.randint(-3, 3))

                # --- 逻辑 B: 处理整数指数 (或已被强制转为整数的) ---
                else:
                    # 如果原值就是整数，检查大小
                    # 注意：如果上面逻辑把 float 强转为了 int，这里就不需要再 check 了
                    # 这里主要处理原本就是 int 但很大的情况
                    if abs(float(target_val)) > 9:
                        target_val = sp.Integer(random.randint(-3, 3))

                new_exp = target_val
            else:
                # 指数是表达式 (如 x**y)，递归处理
                new_exp = _recurse(exp, parent_type=sp.Pow)

            return sp.Pow(new_base, new_exp)

        # ---------------------------
        # 2. 处理普通常数 (Number)
        # ---------------------------
        # 注意：因为上面已经拦截了 is_Pow，这里的 Number 主要是系数、加数等
        if node.is_Number:
            # 检查常数配额
            if context.const_count < 3:
                # 配额未满，保留（但需检查是否越界）
                context.const_count += 1
                clipped = get_clipped_val(node)
                return clipped if clipped is not None else node
            else:
                # 配额已满，需要“消化掉”
                # 如果父节点是乘法，返回 1 (如 3*x*5 -> 3*x*1)
                if parent_type == sp.Mul:
                    return sp.Integer(1)
                # 如果父节点是加法，返回 0 (如 x+5+6 -> x+5+0)
                elif parent_type == sp.Add:
                    return sp.Integer(0)
                # 如果是 Power 的底数或其他情况，返回 1 比较安全 (1**x = 1)
                elif parent_type == sp.Pow:
                    return sp.Integer(1)
                # 默认 fallback
                return sp.Integer(0)

        # ---------------------------
        # 3. 处理其他运算符 (递归遍历)
        # ---------------------------
        if not node.is_Atom:
            # 将当前节点的类型传递给子节点
            # 例如：如果是 Add(x, 5)，5 的 parent_type 就是 Add
            new_args = [_recurse(arg, parent_type=type(node)) for arg in node.args]
            return node.func(*new_args)

        # 4. Symbol 等原子节点直接返回
        return node

    # 启动递归，顶层没有 parent_type
    return _recurse(expr)


###########################################

from sympy import (
    Symbol, Float, Integer, Add, Mul, Pow,
    sin, cos, exp, log, tan, asin, sqrt, Abs,
    sympify, count_ops
)

# -----------------------------------------------------------------------------
# 基础配置 (保持不变)
# -----------------------------------------------------------------------------
UNARY_OPS = [sin, cos, exp, log, tan, asin, sqrt, Abs]
# 简单的黑名单，防止非法数学操作
FORBIDDEN_NESTING = {
    exp: {exp, log}, log: {exp, log},
    sin: {sin, cos, tan, asin},  # 依然尽量避免三角套三角，除非用户输入本身就是这样
    cos: {sin, cos, tan, asin}
}


def is_safe_structure(parent_type, child_node):
    child_type = type(child_node)
    if parent_type in FORBIDDEN_NESTING and child_type in FORBIDDEN_NESTING[parent_type]:
        return False
    return True


def get_vars(expr):
    return {s for s in expr.free_symbols if str(s).startswith('x_')}


# -----------------------------------------------------------------------------
# 激进变异器 (Aggressive Mutator)
# -----------------------------------------------------------------------------

class AggressiveMutator:
    """
    Aggressive expression mutator based on *expression-tree edit operations*.

    Key points (aligned with user constraints):
    - log(x) / sqrt(x) appear as-is (NO Abs/eps, NO (x*x+1) argument substitution).
    - Supports edit-style mutations:
        * subtree insertion / deletion / replacement
        * operator replacement (Add<->Mul, unary swap, Pow exponent tweak)
        * leaf replacement (Symbol / Number)
    - Enforces variable-index contiguity:
        If x_n exists in the mutated expression, then x_{n-1} must also exist.
        (Equivalent: variable indices must form a prefix {1..k}.)
    - Controls growth with max_depth / max_ops and lightweight pruning.
    """

    def __init__(
        self,
        expr,
        max_depth: int = 6,
        max_ops: int = 220,
        max_attempts: int = 12,
        inject_prob: float = 0.55,
    ):
        self.original_expr = expr
        self.max_depth = int(max_depth)
        self.max_ops = int(max_ops)
        self.max_attempts = int(max_attempts)
        self.inject_prob = float(inject_prob)

        # Variable pool is bounded by the maximum index in the *original* expression.
        # This avoids introducing brand-new indices, and contiguity is enforced by validation.
        self.var_pool = self._build_var_pool(expr)

        # Operation probabilities (sum does NOT need to be 1; we sample by ranges)
        self.p_subtree_replace = 0.24
        self.p_subtree_insert = 0.20
        self.p_subtree_delete = 0.14
        self.p_op_replace = 0.16
        self.p_leaf_replace = 0.16
        self.p_const_perturb = 0.10

        # Unary operator palette (tan removed; log/sqrt kept as normal forms)
        # NOTE: exp can also be risky; keep it but lower frequency via weighting.
        self._unary_ops = [sin, cos, log, sqrt, Abs, asin, exp]

        # Risk budget: if too many risky ops already, avoid adding more log/sqrt/exp.
        self.max_risky_ops = 3

    # --------------------------
    # public API
    # --------------------------
    def mutate(self):
        """Perform one mutation with retries; returns a syntactically valid SymPy expression."""
        base = self.original_expr
        best = base

        for _ in range(self.max_attempts):
            expr = base

            # 1) One or two edit operations (tree-level)
            n_edits = 1 if random.random() < 0.65 else 2
            for _k in range(n_edits):
                expr = self._apply_one_edit(expr)

            # 2) Optional top-level noise injection (insert-like)
            if random.random() < self.inject_prob and self._ops_count(expr) < self.max_ops:
                expr = self._inject_noise_term(expr)

            # 3) Global constraints (if available in this script)
            try:
                expr = constrain_expression_values(expr)
            except Exception:
                pass

            # 4) Growth control
            if self._ops_count(expr) > self.max_ops:
                expr = self._prune_expression(expr)

            # 5) Structural validity checks
            if not self._vars_contiguous(expr):
                continue
            if self._violates_forbidden_nesting(expr):
                continue

            # Accept if it actually changed
            if str(expr) != str(base):
                return expr

            best = expr

        # Fallback: force a tiny but nontrivial change that preserves contiguity
        if str(best) == str(base):
            return Add(base, Float(random.uniform(0.01, 0.1)))
        return best

    # --------------------------
    # core: edit operations
    # --------------------------
    def _apply_one_edit(self, expr):
        nodes = self._collect_nodes(expr)
        if not nodes:
            return expr

        # Bias toward non-root edits, but allow root occasionally.
        if len(nodes) > 1 and random.random() < 0.90:
            rec = random.choice(nodes[1:])
        else:
            rec = random.choice(nodes)

        r = random.random()
        if r < self.p_subtree_replace:
            return self._edit_subtree_replace(expr, rec)
        r -= self.p_subtree_replace
        if r < self.p_subtree_insert:
            return self._edit_subtree_insert(expr, rec)
        r -= self.p_subtree_insert
        if r < self.p_subtree_delete:
            return self._edit_subtree_delete(expr, rec)
        r -= self.p_subtree_delete
        if r < self.p_op_replace:
            return self._edit_operator_replace(expr, rec)
        r -= self.p_op_replace
        if r < self.p_leaf_replace:
            return self._edit_leaf_replace(expr, rec)

        return self._edit_constant_perturb(expr, rec)

    def _edit_subtree_replace(self, expr, rec):
        """Replace a subtree with a freshly generated small subtree."""
        new_sub = self._generate_subtree(depth=0)
        return self._replace_at_path(expr, rec["path"], new_sub)

    def _edit_subtree_insert(self, expr, rec):
        """Insert a new term/factor near the chosen node (context-aware when possible)."""
        noise = self._generate_subtree(depth=0, prefer_leaf=True)

        # Try to insert into parent Add/Mul if possible
        if rec["parent"] is not None and rec["parent"].is_Add:
            args = list(rec["parent"].args)
            args.append(noise)
            new_parent = Add(*args)
            return self._replace_at_path(expr, rec["parent_path"], new_parent)

        if rec["parent"] is not None and rec["parent"].is_Mul:
            args = list(rec["parent"].args)
            args.append(noise)
            new_parent = Mul(*args)
            return self._replace_at_path(expr, rec["parent_path"], new_parent)

        # Otherwise wrap the node itself
        if random.random() < 0.60:
            new_node = Add(rec["node"], noise)
        else:
            new_node = Mul(rec["node"], noise)
        return self._replace_at_path(expr, rec["path"], new_node)

    def _edit_subtree_delete(self, expr, rec):
        """Delete a subtree (context-aware neutral deletion for Add/Mul)."""
        # Deleting the root: replace with a leaf to keep expression valid.
        if rec["parent"] is None:
            return self._replace_at_path(expr, rec["path"], self._random_leaf())

        parent = rec["parent"]
        idx = rec["idx_in_parent"]

        # Neutral deletion for Add/Mul
        if parent.is_Add:
            args = list(parent.args)
            if 0 <= idx < len(args):
                args.pop(idx)
            if len(args) == 0:
                new_parent = Integer(0)
            elif len(args) == 1:
                new_parent = args[0]
            else:
                new_parent = Add(*args)
            return self._replace_at_path(expr, rec["parent_path"], new_parent)

        if parent.is_Mul:
            args = list(parent.args)
            if 0 <= idx < len(args):
                args.pop(idx)
            if len(args) == 0:
                new_parent = Integer(1)
            elif len(args) == 1:
                new_parent = args[0]
            else:
                new_parent = Mul(*args)
            return self._replace_at_path(expr, rec["parent_path"], new_parent)

        # Generic: replace subtree with a leaf (keeps syntax)
        return self._replace_at_path(expr, rec["path"], self._random_leaf())

    def _edit_operator_replace(self, expr, rec):
        """Replace operators at the chosen node (operator-level mutation)."""
        node = rec["node"]

        # Add <-> Mul
        if node.is_Add and len(node.args) >= 2:
            return self._replace_at_path(expr, rec["path"], Mul(*node.args))
        if node.is_Mul and len(node.args) >= 2:
            return self._replace_at_path(expr, rec["path"], Add(*node.args))

        # Unary swap: sin/cos/asin, log/sqrt, Abs, exp
        if hasattr(node, "args") and len(node.args) == 1:
            new_func = self._choose_unary(exclude=getattr(node, "func", None), current_expr=expr)
            try:
                return self._replace_at_path(expr, rec["path"], new_func(node.args[0]))
            except Exception:
                return expr

        # Pow exponent tweak
        if node.is_Pow and len(node.args) == 2:
            base, expn = node.args
            new_exp = Integer(random.choice([-2, -1, 2, 3]))
            try:
                return self._replace_at_path(expr, rec["path"], Pow(base, new_exp))
            except Exception:
                return expr

        return expr

    def _edit_leaf_replace(self, expr, rec):
        """Replace a leaf (Symbol/Number) or collapse a small subtree into a leaf."""
        node = rec["node"]

        if node.is_Symbol:
            new_sym = self._random_var(exclude=node)
            return self._replace_at_path(expr, rec["path"], new_sym)

        if node.is_Number:
            if random.random() < 0.55:
                factor = self._sample_nontrivial_float(0.1, 10.0, avoid_center=1.0, avoid_eps=0.18)
                return self._replace_at_path(expr, rec["path"], node * factor)
            delta = self._sample_nontrivial_float(-3, 3, avoid_center=0.0, avoid_eps=0.18)
            return self._replace_at_path(expr, rec["path"], node + delta)

        # If not a leaf, occasionally collapse to a leaf (subtree -> leaf)
        if random.random() < 0.35:
            return self._replace_at_path(expr, rec["path"], self._random_leaf())

        return expr

    def _edit_constant_perturb(self, expr, rec):
        """Perturb a constant somewhere inside; if none, do nothing."""
        node = rec["node"]
        if node.is_Number:
            delta = self._sample_nontrivial_float(-2, 2, avoid_center=0.0, avoid_eps=0.15)
            return self._replace_at_path(expr, rec["path"], node + delta)
        return expr

    # --------------------------
    # injection (outer-layer)
    # --------------------------
    def _inject_noise_term(self, expr):
        term = self._generate_subtree(depth=0, prefer_leaf=False)
        if random.random() < 0.80:
            return Add(expr, term)
        scale = self._sample_nontrivial_float(0.5, 1.5, avoid_center=1.0, avoid_eps=0.10)
        return Mul(scale, expr)

    # --------------------------
    # subtree generation
    # --------------------------
    def _generate_subtree(self, depth=0, prefer_leaf=False):
        """Generate a small subtree to insert/replace, bounded by max_depth."""
        if depth >= min(self.max_depth, 3) or prefer_leaf:
            return self._random_leaf()

        # Slightly bias toward leaves
        if random.random() < 0.35:
            return self._random_leaf()

        choice = random.random()

        # Unary wrapper
        if choice < 0.40:
            op = self._choose_unary(current_expr=None)  # don't need risk check here
            child = self._generate_subtree(depth + 1, prefer_leaf=True)
            try:
                return op(child)
            except Exception:
                return child

        # Binary combine (Add/Mul)
        if choice < 0.85:
            a = self._generate_subtree(depth + 1, prefer_leaf=True)
            b = self._generate_subtree(depth + 1, prefer_leaf=True)
            if random.random() < 0.55:
                return Add(a, b)
            return Mul(a, b)

        # Pow (small integer exponent)
        v = random.choice(self.var_pool) if self.var_pool else Symbol("x_1")
        k = Integer(random.choice([-2, -1, 2, 3]))
        return Pow(v, k)

    # --------------------------
    # validation & structure checks
    # --------------------------
    def _vars_contiguous(self, expr):
        """Ensure variable indices form a prefix {1..k} (no gaps)."""
        indices = self._extract_var_indices(expr)
        if not indices:
            return True
        mx = max(indices)
        want = set(range(1, mx + 1))
        return set(indices) == want

    def _violates_forbidden_nesting(self, expr):
        """Check existing FORBIDDEN_NESTING rules via is_safe_structure."""
        try:
            # Traverse edges parent->child
            stack = [expr]
            while stack:
                node = stack.pop()
                if hasattr(node, "args") and len(node.args) > 0:
                    for child in node.args:
                        try:
                            if not is_safe_structure(getattr(node, "func", None), child):
                                return True
                        except Exception:
                            pass
                        stack.append(child)
        except Exception:
            return False
        return False

    # --------------------------
    # tree utilities
    # --------------------------
    def _collect_nodes(self, expr):
        """Collect nodes with paths and parent info for context-aware edits."""
        out = []

        def rec(node, path, parent=None, parent_path=None, idx_in_parent=None, depth=0):
            out.append(
                {
                    "node": node,
                    "path": path,
                    "parent": parent,
                    "parent_path": parent_path,
                    "idx_in_parent": idx_in_parent,
                    "depth": depth,
                }
            )
            if depth >= self.max_depth:
                return
            args = getattr(node, "args", ())
            for i, ch in enumerate(args):
                rec(ch, path + (i,), parent=node, parent_path=path, idx_in_parent=i, depth=depth + 1)

        rec(expr, ())
        return out

    def _replace_at_path(self, expr, path, new_sub):
        """Purely functional replacement: rebuild only along the path."""
        if not path:
            return new_sub
        node = expr
        i = path[0]
        args = list(getattr(node, "args", ()))
        if not (0 <= i < len(args)):
            return expr
        args[i] = self._replace_at_path(args[i], path[1:], new_sub)
        try:
            return node.func(*args)
        except Exception:
            return expr

    # --------------------------
    # sampling helpers
    # --------------------------
    def _ops_count(self, expr):
        try:
            return int(sp.count_ops(expr, visual=False))
        except Exception:
            return 0

    def _sample_nontrivial_float(self, low, high, avoid_center=None, avoid_eps=0.12, max_tries=20):
        for _ in range(max_tries):
            v = float(random.uniform(low, high))
            if avoid_center is None:
                return Float(v)
            if abs(v - float(avoid_center)) > avoid_eps:
                return Float(v)
        return Float(v)

    def _random_leaf(self):
        # Bias toward variables (keeps expressions meaningful) but still allow constants.
        if self.var_pool and random.random() < 0.60:
            return random.choice(self.var_pool)
        return self._sample_nontrivial_float(-9, 9, avoid_center=0.0, avoid_eps=0.22)

    def _random_var(self, exclude=None):
        pool = self.var_pool if self.var_pool else [Symbol("x_1")]
        candidates = [v for v in pool if v != exclude] or pool
        return random.choice(candidates)

    def _build_var_pool(self, expr):
        vars_ = list(get_vars(expr))
        indices = []
        for v in vars_:
            try:
                m = re.match(r"^x_(\d+)$", str(v))
                if m:
                    indices.append(int(m.group(1)))
            except Exception:
                pass
        mx = max(indices) if indices else 1
        return [Symbol(f"x_{i}") for i in range(1, mx + 1)]

    def _extract_var_indices(self, expr):
        indices = []
        try:
            for v in get_vars(expr):
                m = re.match(r"^x_(\d+)$", str(v))
                if m:
                    indices.append(int(m.group(1)))
        except Exception:
            pass
        return indices

    def _count_risky_ops(self, expr):
        """Count occurrences of risky unary ops and negative exponents."""
        cnt = 0
        try:
            stack = [expr]
            while stack:
                node = stack.pop()
                if getattr(node, "func", None) in (log, sqrt, exp):
                    cnt += 1
                if node.is_Pow and len(node.args) == 2:
                    _b, _e = node.args
                    try:
                        if _e.is_Number and float(_e) < 0:
                            cnt += 1
                    except Exception:
                        pass
                for ch in getattr(node, "args", ()):
                    stack.append(ch)
        except Exception:
            return cnt
        return cnt

    def _choose_unary(self, exclude=None, current_expr=None):
        """Weighted unary selection; optionally respect a risky-op budget."""
        ops = list(self._unary_ops)

        if exclude in ops and len(ops) > 1 and random.random() < 0.85:
            ops = [o for o in ops if o != exclude]

        # Respect risk budget if a current expression is provided
        if current_expr is not None:
            if self._count_risky_ops(current_expr) >= self.max_risky_ops:
                ops = [o for o in ops if o not in (log, sqrt, exp)] or ops

        # Weighted choice: log/sqrt/exp less frequent
        weights = []
        for op in ops:
            if op in (log, sqrt, exp):
                weights.append(0.6)
            elif op is Abs:
                weights.append(1.2)
            else:
                weights.append(1.0)

        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for op, w in zip(ops, weights):
            acc += w
            if r <= acc:
                return op
        return ops[-1]

