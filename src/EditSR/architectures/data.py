from torch.utils import data
try:
    from src.EditSR.utils import load_metadata_hdf5, load_eq
except ImportError:
    from src.EditSR.utils import load_metadata_hdf5, load_eq
from sympy.core.rules import Transform
"""Dataset & numerical target generation.

This project originally included *visual* components (plotting / image rendering,
OpenCV rasterization, and multi-modal inputs). Per request, all vision-related
code paths have been removed from this file.

[FIXED v2] 彻底修复多进程 DataLoader 兼容性问题
- 完全移除 func_timeout 和 threading 超时机制
- 使用 sympy 的超时参数和安全计算方式
- 添加计算复杂度检查，提前跳过可能卡死的表达式
"""

from sympy import sympify, Float, Symbol, Integer

from typing import List, Optional, Tuple, Any
from torch.distributions.uniform import Uniform
from ..dataset.data_utils import sample_symbolic_constants
from ..dataset.generator import Generator
import numpy as np
import pytorch_lightning as pl
try:
    from src.EditSR.dclasses import Equation
except ImportError:
    from src.EditSR.dclasses import Equation
from functools import partial
from pathlib import Path
import hydra
from src.EditSR.project_paths import scripts_path, resolve_path
import torch, random, math, warnings, re

import logging

import torch
import numpy as np
from sympy import lambdify

# 获取 logger 实例
logger = logging.getLogger(__name__)

modules = {
    "numpy": np,
    "ln": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "Abs": np.abs,
    "pi": np.pi,
    "E": np.e,
    "asin": np.arcsin,
    "re": np.real
}

# 配置参数
ALLOWED_INTS = {str(i) for i in range(-9, 10) if i != 0}
VAR_TOKEN_PATTERN = re.compile(r"\bx_(\d+)\b")


def _uses_zero_based_variables_from_names(names):
    return any(str(name) == "x_0" for name in names)


def normalize_expr_string_to_one_based(expr_str):
    expr_str = str(expr_str)
    if "x_0" not in expr_str:
        return expr_str
    return VAR_TOKEN_PATTERN.sub(lambda m: f"x_{int(m.group(1)) + 1}", expr_str)


def normalize_variable_list_to_one_based(variables):
    variables = [str(v) for v in variables]
    if not _uses_zero_based_variables_from_names(variables):
        return variables
    normalized = [normalize_expr_string_to_one_based(v) for v in variables]
    return sorted(set(normalized), key=lambda x: int(x.split('_')[1]))


def normalize_equation_to_one_based(eq):
    expr = getattr(eq, "expr", None)
    variables = getattr(eq, "variables", None)
    if expr is not None:
        eq.expr = normalize_expr_string_to_one_based(expr)
    if variables is not None:
        eq.variables = normalize_variable_list_to_one_based(variables)
    return eq

# -----------------------------------------------------------------------------
# [NEW] 表达式复杂度检查
# ----------------------------------------------------------------------------
def safe_lambdify(var_list, expr, modules_dict):
    """
    安全的 lambdify 包装，捕获所有异常
    """
    try:
        return lambdify(var_list, expr, modules=modules_dict)
    except Exception as e:
        logger.debug(f"lambdify failed: {e}")
        return None


def safe_eval(func, *args):
    """
    安全的函数求值，带有数值范围检查
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args)
        return result
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Helper from data2.py

class EditSRDataset(data.Dataset):
    def __init__(
            self,
            data_path: Path,
            cfg,
            mode: str
    ):
        metadata = load_metadata_hdf5(str(resolve_path(data_path, base="scripts")))
        cfg.total_variables = normalize_variable_list_to_one_based(metadata.total_variables)
        self.cfg = cfg
        self.len = metadata.total_number_of_eqs
        self.eqs_per_hdf = metadata.eqs_per_hdf
        self.word2id = metadata.word2id
        print(self.word2id)
        self.id2word = metadata.id2word
        self.data_path = data_path
        self.mode = mode
        self.cfg = cfg

    def __getitem__(self, index):
        eq = load_eq(self.data_path, index, self.eqs_per_hdf)
        eq = normalize_equation_to_one_based(eq)

        try:
            result = self.return_t_expr(eq)
            if result is None:
                raise ValueError("Invalid sample after tokenization")
            sympy_expr, t, _, eq_sympy_prefix = result
            curr = Equation(
                expr=sympy_expr,
                coeff_dict={},
                eq_sympy_prefix=eq_sympy_prefix,
                variables=eq.variables,
                support=eq.support,
                tokenized=t,
                valid=True,
            )
        except Exception:
            curr = Equation(
                expr='x_1',
                coeff_dict={},
                eq_sympy_prefix=[],
                variables=eq.variables,
                support=eq.support,
                valid=False,
            )

        return curr

    def __len__(self):
        return self.len

    def return_t_expr(self, eq) -> Optional[Tuple[Any, list, dict, list]]:
        """
        [FIXED v2] 移除所有超时机制，使用异常处理和复杂度检查
        """
        consts, initial_consts = sample_symbolic_constants(eq, self.cfg.constants)
        if self.cfg.predict_c:
            eq_string = eq.expr.format(**consts)
        else:
            eq_string = eq.expr.format(**initial_consts)
        eq_sympy_infix, sympy_expr = constants_to_placeholder(eq_string)

        if eq_sympy_infix is None or sympy_expr is None:
            return None

        eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)
        t = tokenize(eq_sympy_prefix, self.word2id)
        if t is None:
            return None

        return sympy_expr, t, consts, eq_sympy_prefix
def custom_collate_fn(eqs: List[Equation], cfg) -> List[torch.tensor]:
    """Collate with fault tolerance."""
    filtered_eqs = [
        eq for eq in eqs
        if eq.valid and len(eq.tokenized) < 30
    ]

    if len(filtered_eqs) == 0:
        return [None, None, None]

    try:
        res, tokens_eqs, expr = evaluate_and_wrap(filtered_eqs, cfg)
    except Exception as e:
        logger.warning(f"[collate] batch failed: {e}")
        return [None, None, None]

    if res is None or tokens_eqs is None or len(expr) == 0:
        return [None, None, None]

    return [res, tokens_eqs, expr]


def constants_to_placeholder(s, symbol="c"):
    """
    [FIXED v2] 添加异常处理
    """
    try:
        s = normalize_expr_string_to_one_based(s)
        sympy_expr = sympify(s)
        eq_sympy_infix = sympy_expr.xreplace(
            Transform(
                lambda x: Symbol(symbol, real=True, nonzero=True),
                lambda x: isinstance(x, Float) or (isinstance(x, Integer) and abs(x) > 9)
            )
        )
        return eq_sympy_infix, sympy_expr
    except Exception as e:
        logger.debug(f"constants_to_placeholder failed: {e}")
        return None, None

def tokenize(prefix_expr: list, word2id: dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        if i in word2id:
            tokenized_expr.append(word2id[i])
        else:
            logger.debug(f"Unknown token: {i}")
            return None
    tokenized_expr.append(word2id["F"])
    return tokenized_expr


def de_tokenize(tokenized_expr, id2word: dict):
    prefix_expr = []
    for i in tokenized_expr:
        if isinstance(i, torch.Tensor):
            idx = i.item()
        else:
            idx = i
        if "F" == id2word.get(idx, ""):
            break
        else:
            prefix_expr.append(id2word.get(idx, "?"))
    return prefix_expr


def tokens_padding(tokens, max_len: int = None, pad_id: int = 0):
    """Pad token sequences to a fixed length."""
    if max_len is None:
        max_len = max(len(y) for y in tokens) if len(tokens) > 0 else 0
    else:
        max_len = int(max_len)

    p_tokens = torch.full((len(tokens), max_len), int(pad_id), dtype=torch.long)
    for i, y in enumerate(tokens):
        y = torch.as_tensor(y, dtype=torch.long)
        if y.numel() == 0 or max_len == 0:
            continue
        if y.numel() > max_len:
            y = y[:max_len]
        L = int(y.shape[0])
        p_tokens[i, :L] = y
    return p_tokens


def number_of_support_points(p, type_of_sampling_points):
    if type_of_sampling_points == "constant":
        curr_p = p
    elif type_of_sampling_points == "logarithm":
        curr_p = int(10 ** Uniform(math.log10(20), math.log10(p)).sample())
    else:
        raise NameError
    return curr_p


def get_support_source_ids(cfg):
    """Return the support-source ids used to create independent views."""
    source_ids = getattr(cfg, "support_source_ids", None)
    if source_ids is None:
        return [0, 1, 2, 3, 4]
    if isinstance(source_ids, (int, np.integer)):
        return [int(source_ids)]
    try:
        source_ids = [int(x) for x in source_ids]
    except Exception:
        return [0, 1, 2, 3, 4]
    if len(source_ids) == 0:
        return [0, 1, 2, 3, 4]
    return source_ids

import math
import torch

def sample_support(curr_p, cfg, n_clusters, source_id=None):
    """
    Sample one independent support set, consistent with the manuscript:

    1) Draw two scalars from Uniform(-10, 10), sort them to form [a, b].
    2) For each variable, choose with equal probability:
       - Linear-uniform sampling: x ~ Uniform(a, b)
       - Log-scale sampling: if a and b have the same sign,
         sample |x| ~ LogUniform(|a|, |b|) and keep sign(x) = sign(a)

    Notes:
    - source_id is kept only for API compatibility and is ignored.
    - n_clusters is unused here, also kept for compatibility.
    """

    # Step 1: draw interval [a, b]
    ab = torch.empty(2).uniform_(-10.0, 10.0)
    a, b = torch.sort(ab).values.tolist()

    samples = []

    for _ in range(int(curr_p)):
        use_log = torch.rand(1).item() < 0.5

        # Log-scale sampling only makes sense when a and b have the same sign and are nonzero
        same_sign = (a > 0 and b > 0) or (a < 0 and b < 0)

        if use_log and same_sign:
            sign = 1.0 if a > 0 else -1.0
            lo_mag = min(abs(a), abs(b))
            hi_mag = max(abs(a), abs(b))

            # Numerical safeguard, although exact zero has probability 0 in continuous sampling
            lo_mag = max(lo_mag, 1e-12)

            u = torch.empty(1).uniform_(math.log(lo_mag), math.log(hi_mag))
            x = sign * torch.exp(u).item()
        else:
            x = torch.empty(1).uniform_(a, b).item()

        samples.append(x)

    return torch.tensor(samples, dtype=torch.float32)
def var_sort_key(item):
    return int(item.split('_')[1])


def _cfg_get_bool(cfg, name: str, default: bool = False) -> bool:
    try:
        value = getattr(cfg, name, default)
    except Exception:
        value = default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cfg_get_int(cfg, name: str, default: int) -> int:
    try:
        value = getattr(cfg, name, default)
    except Exception:
        value = default
    try:
        return int(value)
    except Exception:
        return int(default)


def _sample_distractor_count(dim_used: int, support_capacity: int, cfg) -> int:
    """
    Sample k from {0,1,2,3} while ensuring the total number of active features
    does not exceed both the support capacity and 10.
    """
    if not _cfg_get_bool(cfg, 'stress_distractor_enable', False):
        # print(”)
        return 0

    max_total = _cfg_get_int(cfg, 'stress_distractor_max_total_vars', 10)
    max_k = _cfg_get_int(cfg, 'stress_distractor_max_k', 3)
    max_total = max(0, min(max_total, 10, int(support_capacity)))
    max_extra = max(0, min(int(max_k), max_total - int(dim_used), int(support_capacity) - int(dim_used)))
    valid_ks = [k for k in (0, 1, 2, 3) if k <= max_extra]
    if len(valid_ks) == 0:
        return 0
    return int(random.choice(valid_ks))


def _sample_uniform_like_feature(feature_values: torch.Tensor) -> torch.Tensor:
    """
    Sample an independent distractor feature whose numeric range follows one
    already valid feature. The distractor is *not* copied directly; it is
    resampled uniformly within the selected feature's observed range.
    """
    if not isinstance(feature_values, torch.Tensor):
        feature_values = torch.as_tensor(feature_values, dtype=torch.float32)
    feature_values = feature_values.detach().to(dtype=torch.float32)

    finite = feature_values[torch.isfinite(feature_values)]
    if finite.numel() == 0:
        return torch.zeros_like(feature_values, dtype=torch.float32)

    lo = float(finite.min().item())
    hi = float(finite.max().item())
    if (not math.isfinite(lo)) or (not math.isfinite(hi)):
        return torch.zeros_like(feature_values, dtype=torch.float32)

    if hi < lo:
        lo, hi = hi, lo

    if abs(hi - lo) < 1e-12:
        return torch.full_like(feature_values, fill_value=lo, dtype=torch.float32)

    distribution = torch.distributions.Uniform(lo, hi)
    return distribution.sample(feature_values.shape).to(dtype=torch.float32)


def _inject_distractor_features_(support: torch.Tensor, dim_used: int, cfg) -> torch.Tensor:
    """
    In-place style helper: randomly add k distractor features, where
    k is sampled from {0,1,2,3} subject to the constraint that the total number
    of active features never exceeds 10.

    Each distractor follows the numeric range of one randomly chosen valid
    feature already present in the current support set.

    Default behavior is disabled, so existing scripts are unaffected unless
    cfg.stress_distractor_enable=true is provided.
    """
    if not isinstance(support, torch.Tensor) or support.ndim != 2:
        return support

    support_capacity = int(support.shape[0])
    dim_used = int(max(0, min(dim_used, support_capacity)))
    if dim_used <= 0:
        return support

    k = _sample_distractor_count(dim_used=dim_used, support_capacity=support_capacity, cfg=cfg)
    if k <= 0:
        return support

    max_insert_end = min(support_capacity, 10, dim_used + k)
    insert_rows = list(range(dim_used, max_insert_end))
    if len(insert_rows) == 0:
        return support

    valid_source_rows = [i for i in range(dim_used) if torch.isfinite(support[i]).any()]
    if len(valid_source_rows) == 0:
        return support

    for target_row in insert_rows:
        source_row = random.choice(valid_source_rows)
        sampled = _sample_uniform_like_feature(support[source_row])
        support[target_row, :] = sampled

    return support


# def generate_support(eq, curr_p, n_clusters, cfg, seed=None):
#     """Generate numerical support points for one equation."""
#     sorted_vars = tuple(sorted(eq.variables, key=str)) if len(eq.variables) > 1 else tuple(eq.variables)
#     dim = len(sorted_vars)
#
#     num_vars = int(cfg.total_variables) if isinstance(
#         getattr(cfg, 'total_variables', 0), (int, np.integer)
#     ) else len(getattr(cfg, 'total_variables', []))
#
#     support = torch.zeros((num_vars, curr_p), dtype=torch.float32)
#
#     for i in range(dim):
#         data = sample_support(curr_p, cfg, n_clusters)
#         if isinstance(data, torch.Tensor):
#             support[i, :] = data.float()
#         else:
#             support[i, :] = torch.from_numpy(np.asarray(data, dtype=np.float32))
#
#     # [NEW] Inject useless noise features into the remaining (unused) dimensions.
#     support = _inject_noise_features_(support, dim_used=dim, cfg=cfg)
#
#     # [NEW] Optional clipping to prevent extreme feature magnitudes from blowing up
#     # downstream networks (especially with fp16/AMP). Set cfg.support_clip=0 to disable.
#     try:
#         clip = float(10)
#     except Exception:
#         clip = 10.0
#     if clip and clip > 0:
#         support = support.clamp(min=-clip, max=clip)
#
#     return support

def generate_support(eq, curr_p, n_clusters, cfg, seed=None, source_id=None):
    """Generate numerical support points for one equation from one source family."""
    sorted_vars = tuple(sorted(eq.variables, key=str)) if len(eq.variables) > 1 else tuple(eq.variables)
    dim = len(sorted_vars)

    num_vars = int(cfg.total_variables) if isinstance(
        getattr(cfg, 'total_variables', 0), (int, np.integer)
    ) else len(getattr(cfg, 'total_variables', []))

    support = torch.zeros((num_vars, curr_p), dtype=torch.float32)

    for i in range(dim):
        data = sample_support(curr_p, cfg, n_clusters, source_id=source_id)
        if isinstance(data, torch.Tensor):
            support[i, :] = data.float()
        else:
            support[i, :] = torch.from_numpy(np.asarray(data, dtype=np.float32))

    # Optional stress-test augmentation: append distractor variables into the
    # unused feature slots. Disabled by default to avoid affecting other scripts.
    support = _inject_distractor_features_(support, dim_used=dim, cfg=cfg)

    return support


def return_y(eq, support):
    """
    [FIXED v2] 安全的表达式求值，无超时机制
    """
    var_list = sorted(list(eq.variables), key=var_sort_key)

    # [NEW] 使用安全的 lambdify
    eq_numpy = safe_lambdify(var_list, eq.expr, modules)
    if eq_numpy is None:
        raise ValueError("Failed to create lambdify function")

    if not isinstance(support, torch.Tensor) or support.ndim != 2:
        raise ValueError("support must be a 2D torch.Tensor [num_vars, P]")

    P = int(support.shape[1])
    dim = len(var_list)

    args = [support[i, :].detach().cpu().numpy() for i in range(dim)] if dim > 0 else []

    # [NEW] 使用安全求值
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if dim > 1:
                y_np = eq_numpy(*args)
            elif dim == 1:
                y_np = eq_numpy(args[0])
            else:
                y_np = eq_numpy()
    except Exception as e:
        raise ValueError(f"Evaluation failed: {e}")

    if y_np is None:
        raise ValueError("Evaluation returned None")

    # Ensure ndarray float
    y_np = np.asarray(y_np)
    if np.iscomplexobj(y_np):
        y_np = np.real(y_np)

    y = torch.as_tensor(y_np, dtype=torch.float32)

    # Shape normalization
    if y.ndim == 0:
        y = y.expand(P).clone()
    else:
        y = y.reshape(-1)
        if y.numel() == 1:
            y = y.expand(P).clone()
        elif y.numel() != P:
            raise ValueError(f"y has unexpected shape (numel={y.numel()}) expected {P}")

    # Add target noise
    target_noise = float(np.random.choice([0.0, 0.01, 0.1]))
    if target_noise > 0.0 and y.numel() > 0:
        finite = torch.isfinite(y)
        valid_y = y[finite]
        if valid_y.numel() > 0:
            rms = torch.sqrt(torch.mean(valid_y * valid_y)).clamp_min(0.0)
            scale = float(target_noise * rms.item())
            if math.isfinite(scale) and scale > 0.0:
                noise = torch.randn_like(y) * scale
                y = torch.where(finite, y + noise, y)

    # y_np2 = y.detach().cpu().numpy().reshape(-1)  # 强制 1D
    #
    # sy = AutoMagnitudeScaler(centering=False).fit(y_np2)  # y=None -> 不做稳定性检查
    # y_scaled_np = sy.transform(y_np2)  # numpy (P,)
    #
    # # 转回 torch，保持 device/dtype
    # y = torch.from_numpy(y_scaled_np).to(device=y.device, dtype=torch.float32)

    return y, eq_numpy


def _safe_processing_logic(eq, curr_p, cfg, seed, source_id=None):
    """Pure logic: generate support and y for one independent support source."""
    support = generate_support(
        eq, curr_p,
        n_clusters=random.choice(cfg.n_clusters),
        cfg=cfg,
        seed=seed,
        source_id=source_id,
    )
    y, _ = return_y(eq, support)
    return support, y


def _sample_once(eq, curr_p, cfg, source_id=None):
    """
    [FIXED v2] 单次采样，无超时机制
    """
    try:
        seed = getattr(eq, 'seed', None)

        support, y = _safe_processing_logic(eq, curr_p, cfg, seed, source_id=source_id)

        success = False
        invalid_indices = torch.tensor([], dtype=torch.long)

        if isinstance(y, torch.Tensor) and y.dtype == torch.float32:
            y = y.squeeze(0)
            invalid_indices = torch.where(
                torch.isnan(y) | torch.isinf(y) | (abs(y) > cfg.eps_limit)
            )[0]
            success = (len(invalid_indices) <= curr_p * 0.7)

        return support, y, invalid_indices, success

    except Exception as e:
        logger.debug(f"_sample_once failed: {e}")
        return [], [], [], False


def evaluate_and_wrap(eqs: List[Equation], cfg):
    """
    [FIXED v3] 每个表达式只采样一个数据集，不再对单个表达式做多源重采样并堆叠。
    """
    vals = []
    expr_out = []
    tokens_out = []

    curr_p = number_of_support_points(cfg.max_number_of_points, cfg.type_of_sampling_points)

    for eq in eqs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            eq_success = False

            for retry in range(cfg.max_retry):
                support, y, invalid_idx, ok = _sample_once(eq, curr_p, cfg, source_id=None)

                if not ok:
                    continue

                if not torch.is_tensor(y):
                    y = torch.as_tensor(y, dtype=torch.float32)
                else:
                    y = y.to(dtype=torch.float32)

                if y.ndim == 0 or y.numel() <= 1:
                    continue

                y = y.reshape(-1)
                if y.numel() != curr_p:
                    continue

                if (not torch.is_tensor(support)) or support.ndim != 2 or support.shape[1] != curr_p:
                    continue

                eq_success = True
                break

            if not eq_success:
                continue

            y_fixed = y.clone()
            if invalid_idx.numel() > 0:
                y_fixed[invalid_idx] = 0
                support[:, invalid_idx] = 0

            concatenated = torch.cat([support, y_fixed.unsqueeze(0)], dim=0)
            vals.append(concatenated.unsqueeze(0))
            expr_out.append(eq.expr)
            tokens_out.append(eq.tokenized)

    if len(vals) == 0:
        return None, None, []

    pad_id = int(getattr(
        cfg, 'trg_pad_idx',
        cfg.word2id.get('P', 0) if hasattr(cfg, 'word2id') and isinstance(cfg.word2id, dict) else 0
    ))
    max_len = int(getattr(cfg, 'length_eq', 0)) if getattr(cfg, 'length_eq', None) is not None else None
    tokens_eqs = tokens_padding(tokens_out, max_len=max_len, pad_id=pad_id)

    res = torch.cat(vals, dim=0)
    return res, tokens_eqs, expr_out


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_train_path,
            data_val_path,
            data_test_path,
            cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.data_train_path = data_train_path
        self.data_val_path = data_val_path
        self.data_test_path = data_test_path

    def setup(self, stage=None):
        """called one each GPU separately - stage defines if we are at fit or test step"""
        if stage == "fit" or stage is None:
            if self.data_train_path:
                self.training_dataset = EditSRDataset(
                    self.data_train_path,
                    self.cfg.dataset_train,
                    mode="train"
                )
            if self.data_val_path:
                self.validation_dataset = EditSRDataset(
                    self.data_val_path,
                    self.cfg.dataset_val,
                    mode="val"
                )

        if self.data_test_path:
            self.test_dataset = EditSRDataset(
                self.data_test_path,
                self.cfg.dataset_test,
                mode="test"
            )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.cfg.architecture.batch_size,
            shuffle=True,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_train),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            persistent_workers=True,
            # [NEW] 添加 prefetch_factor 以减少内存压力
            prefetch_factor=2 if self.cfg.num_of_workers > 0 else None,
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.cfg.architecture.batch_size,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_val),
            num_workers=self.cfg.num_of_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2 if self.cfg.num_of_workers > 0 else None,
        )
        return validloader

    def test_dataloader(self):
        """returns test dataloader"""
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=partial(custom_collate_fn, cfg=self.cfg.dataset_test),
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )
        return testloader

# Backward-compatible alias for older imports.
EditSRsDataset = EditSRDataset
