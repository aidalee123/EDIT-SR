"""Evaluate distractor robustness and print per-task results directly to stdout."""

import os
import sys
import time
import gc
import re
import json
import types
import random
import warnings
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import hydra
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import sympy as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sympy import lambdify, sympify, preorder_traversal

# --- 环境设置 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from src.EditSR.architectures.model import Model
from src.EditSR.dclasses import BFGSParams, FitParams
from src.EditSR.utils import load_metadata_hdf5, symbol_equivalence_single, AutoMagnitudeScaler
from src.EditSR.project_paths import scripts_path, resolve_path

try:
    from src.EditSR.architectures import model as model_module
except Exception:
    import model as model_module

try:
    from src.EditSR.architectures.data import (
        tokenize as project_tokenize,
        constants_to_placeholder as project_constants_to_placeholder,
        normalize_expr_string_to_one_based,
    )
except Exception:
    try:
        from data import (
            tokenize as project_tokenize,
            constants_to_placeholder as project_constants_to_placeholder,
            normalize_expr_string_to_one_based,
        )
    except Exception:
        project_tokenize = None
        project_constants_to_placeholder = None

        def normalize_expr_string_to_one_based(expr_str):
            return str(expr_str)

from src.EditSR.dataset.generator import Generator as ProjectGenerator
warnings.filterwarnings("ignore")


# ==========================================
# 通用工具
# ==========================================

def resolve_first_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError(f"未找到文件，已尝试: {candidates}")


def calculate_tree_size(expression_str):
    try:
        expr = sp.sympify(str(expression_str))
        return sum(1 for _ in preorder_traversal(expr))
    except Exception:
        return -1


def get_zero_dims_from_x(X):
    zero_columns = np.all(X == 0, axis=0)
    zero_var_names = [f'x_{i + 1}' for i, is_zero in enumerate(zero_columns) if is_zero]
    return zero_var_names


def round_if_needed(val):
    num = float(val)
    rounded = round(num, 1)
    if abs(rounded - int(rounded)) < 1e-10:
        return sp.Integer(int(rounded))
    return sp.Float(rounded)


def process_expr(expr, in_exponent=False):
    if expr.is_Atom:
        if expr.is_number and expr.free_symbols == set() and not in_exponent:
            try:
                return round_if_needed(expr)
            except Exception:
                return expr
        return expr
    if expr.func == sp.Pow:
        base = process_expr(expr.args[0])
        exponent = process_expr(expr.args[1])
        return sp.Pow(base, exponent)
    new_args = tuple(process_expr(arg, in_exponent=False) for arg in expr.args)
    return expr.func(*new_args)


# 兼容旧逻辑，保留但扩展到更多变量
_XY_ZTV_MAP = {
    r'\bx\b': 'x_1',
    r'\by\b': 'x_2',
    r'\bz\b': 'x_3',
    r'\bt\b': 'x_4',
    r'\bv\b': 'x_5',
    r'\bu\b': 'x_6',
    r'\bw\b': 'x_7',
    r'\bs\b': 'x_8',
    r'\br\b': 'x_9',
    r'\bq\b': 'x_10',
}


def replace_variables(expression):
    expr = str(expression)
    for pat, rep in _XY_ZTV_MAP.items():
        expr = re.sub(pat, rep, expr)
    return expr


_SUM_HARMONIC_PAT = re.compile(r"sum_\{i=1\}\^\{x_1\}\s*\(1/i\)", re.IGNORECASE)
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_UE_PAT = re.compile(rf"^\s*([UE])\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*(\d+)\s*\]\s*$")


def process_benchmark_expression(expression_raw: str) -> str:
    """
    面向 table3_with_n200.csv 的表达式预处理。
    与 rils-rols_std.py 的语法兼容，但保留 x_1, x_2, ... 变量命名。
    """
    expression = str(expression_raw).strip()

    if expression.startswith("'"):
        expression = expression[1:].strip()

    expression = re.sub(r"\s*\(i\.e\..*$", "", expression).strip()
    expression = re.sub(r"pow\((.*?),(.*?)\)", r"((\1) ** (\2))", expression)
    expression = re.sub(r"div\((.*?),(.*?)\)", r"((\1) / (\2))", expression)
    expression = expression.replace("^", "**")
    expression = re.sub(r"\bln\s*\(", "log(", expression)
    expression = re.sub(r"\barcsinh\b", "asinh", expression)
    expression = _SUM_HARMONIC_PAT.sub("harmonic(x_1)", expression)

    # 兼容 x1/x2/... -> x_1/x_2/...
    expression = re.sub(r"\bx(\d+)\b", lambda m: f"x_{m.group(1)}", expression)
    return expression


# ==========================================
# token / trace / distance 辅助函数
# ==========================================

def safe_to_list(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return list(x)



def trim_token_sequence(seq_like, pad_id: int, finish_id: int, include_finish: bool = True):
    seq = [int(v) for v in safe_to_list(seq_like)]
    seq = [v for v in seq if v != int(pad_id)]
    if int(finish_id) in seq:
        fidx = seq.index(int(finish_id))
        seq = seq[:fidx + 1] if include_finish else seq[:fidx]
    return seq



def token_ids_to_words(token_ids, id2word):
    return [str(id2word.get(int(t), f"<{int(t)}>")) for t in token_ids]



def pack_token_sequence(seq_like, pad_id: int, start_id: int, finish_id: int, max_len: int, device):
    seq = [int(v) for v in safe_to_list(seq_like)]
    seq = [v for v in seq if v != int(pad_id)]
    if int(finish_id) in seq:
        seq = seq[:seq.index(int(finish_id))]
    if len(seq) == 0 or seq[0] != int(start_id):
        seq = [int(start_id)] + seq
    packed = seq + [int(finish_id)]
    packed = packed[:max_len]
    if int(finish_id) not in packed:
        packed[-1] = int(finish_id)
    out = torch.full((1, max_len), int(pad_id), dtype=torch.long, device=device)
    out[0, :len(packed)] = torch.tensor(packed, dtype=torch.long, device=device)
    return out



def token_to_key(tok_like):
    return tuple(int(v) for v in safe_to_list(tok_like))



def levenshtein_distance(seq_a, seq_b):
    a = list(seq_a)
    b = list(seq_b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, aa in enumerate(a, start=1):
        cur = [i]
        for j, bb in enumerate(b, start=1):
            cost = 0 if aa == bb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return int(prev[-1])



def normalized_edit_distance(seq_a, seq_b):
    denom = max(len(seq_a), len(seq_b), 1)
    return float(levenshtein_distance(seq_a, seq_b) / denom)



def json_dumps_safe(obj):
    return json.dumps(obj, ensure_ascii=False)



def tokenize_true_expression(true_expr: str, word2id: dict, max_len: int, pad_id: int, finish_id: int):
    if project_tokenize is None or project_constants_to_placeholder is None or ProjectGenerator is None:
        raise RuntimeError("无法导入项目内的 tokenization 工具，请检查 src.EditSR.data 与 src.EditSR.dataset.generator 是否可用。")

    expr_norm = normalize_expr_string_to_one_based(str(true_expr))
    eq_sympy_infix, _ = project_constants_to_placeholder(expr_norm)
    if eq_sympy_infix is None:
        raise ValueError(f"true_expr 无法 token 化: {true_expr}")

    prefix = ProjectGenerator.sympy_to_prefix(eq_sympy_infix)
    tok = project_tokenize(prefix, word2id)
    if tok is None:
        raise ValueError(f"true_expr tokenization 失败: {true_expr}")

    tok = [int(x) for x in tok]
    if len(tok) > max_len:
        tok = tok[:max_len]
        if int(finish_id) not in tok:
            tok[-1] = int(finish_id)
    else:
        tok = tok + [int(pad_id)] * (max_len - len(tok))
    return tok



def install_encoder_tap(model):
    encoder = model.MultiModalEncoder
    tap = {'last_output': None}
    original_forward = encoder.forward

    def wrapped_forward(self_module, *args, **kwargs):
        out = original_forward(*args, **kwargs)
        try:
            tap['last_output'] = out.detach().clone()
        except Exception:
            tap['last_output'] = out
        return out

    encoder.forward = types.MethodType(wrapped_forward, encoder)
    return tap, original_forward



def uninstall_encoder_tap(model, original_forward):
    model.MultiModalEncoder.forward = original_forward



def compute_trace_train_r2(expr_str, X_cpu, y_cpu, test_data):
    if expr_str is None:
        return float('nan')
    try:
        expr = sp.sympify(str(expr_str))
        y_true = np.asarray(y_cpu.detach().cpu().numpy()).reshape(-1)
        values = {x: X_cpu[:, :, idx].detach().cpu().numpy() for idx, x in enumerate(test_data.total_variables)}
        y_found = sp.lambdify(",".join(test_data.total_variables), expr, modules=model_module.bfgs.modules)(**values)
        y_pred = np.asarray(y_found)
        if y_pred.ndim == 0:
            y_pred = np.full_like(y_true, float(y_pred), dtype=np.float64)
        else:
            y_pred = y_pred.reshape(-1)
        if y_pred.shape[0] != y_true.shape[0]:
            y_pred = np.broadcast_to(y_pred, y_true.shape).copy()
        if not np.all(np.isfinite(y_pred)):
            return float('nan')
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float('nan')



def run_bfgs_on_tokens(token_list, X_cpu, y_cpu, cfg_params, test_data, eval_seed: int):
    bfgs_enabled = bool(getattr(cfg_params.bfgs, 'activated', True))
    if (not bfgs_enabled) or len(token_list) == 0:
        return {}

    max_workers = int(max(1, int(getattr(cfg_params.bfgs, 'max_workers', 20))))
    task_list = [
        (tok, X_cpu, y_cpu, cfg_params, test_data, int(eval_seed + i))
        for i, tok in enumerate(token_list)
    ]

    eval_by_key = {}
    with ProcessPoolExecutor(max_workers) as executor:
        futures = [executor.submit(model_module.bfgs_wrapper, args) for args in task_list]
        for future, tok in zip(futures, token_list):
            pred_w_c, loss_bfgs, ww_ = future.result()
            if pred_w_c is None:
                continue
            eval_by_key[token_to_key(ww_)] = {
                'pred_expr': pred_w_c,
                'bfgs_loss': float(loss_bfgs),
                'bfgs_r2': float(compute_trace_train_r2(pred_w_c, X_cpu, y_cpu, test_data)),
                'token': ww_,
            }
    return eval_by_key



def better_record(a, b):
    if b is None:
        return False
    if a is None:
        return True
    ar2 = float(a.get('bfgs_r2', float('nan')))
    br2 = float(b.get('bfgs_r2', float('nan')))
    aloss = float(a.get('bfgs_loss', float('nan')))
    bloss = float(b.get('bfgs_loss', float('nan')))
    akey = (
        1 if np.isfinite(ar2) else 0,
        ar2 if np.isfinite(ar2) else -1.0e18,
        1 if np.isfinite(aloss) else 0,
        -aloss if np.isfinite(aloss) else -1.0e18,
        float(a.get('score', -1.0e18)),
    )
    bkey = (
        1 if np.isfinite(br2) else 0,
        br2 if np.isfinite(br2) else -1.0e18,
        1 if np.isfinite(bloss) else 0,
        -bloss if np.isfinite(bloss) else -1.0e18,
        float(b.get('score', -1.0e18)),
    )
    return bkey > akey



def empty_repair_trace_dict(max_iters: int):
    out = {
        'repair_trace_source_beam': np.nan,
        'repair_trace_reason': '',
        'repair_trace_skipped': 0,
        'repair_trace_executed_iters': 0,
        'repair_trace_early_stop_hit': 0,
        'repair_trace_early_stop_iter': np.nan,
        'repair_trace_true_token_ids': json_dumps_safe([]),
        'repair_trace_true_tokens': json_dumps_safe([]),
    }
    for i in range(1, max_iters + 1):
        out[f'repair_iter_{i}_state_count'] = 0
        out[f'repair_iter_{i}_avg_norm_edit_distance'] = np.nan
        out[f'repair_iter_{i}_state_token_ids'] = json_dumps_safe([])
        out[f'repair_iter_{i}_state_tokens'] = json_dumps_safe([])
    return out



def collect_repair_trace_stats(
    model,
    output,
    src_enc,
    X_tensor,
    y_tensor,
    cfg_params,
    test_data,
    true_expr,
    source_beam,
):
    max_iters = int(getattr(cfg_params, 'repair_max_iters', 10))
    trace = empty_repair_trace_dict(max_iters)
    trace['repair_trace_source_beam'] = int(source_beam)

    if src_enc is None:
        trace['repair_trace_reason'] = 'missing_src_enc'
        return trace

    if bool(output.get('repair_skipped_due_to_perfect_baseline', False)):
        trace['repair_trace_reason'] = 'repair_skipped_due_to_perfect_baseline'
        trace['repair_trace_skipped'] = 1
        return trace

    seed_tokens_raw = output.get('repair_input_tokens', []) or []
    seed_tokens_raw = [tok for tok in seed_tokens_raw if tok is not None]
    if len(seed_tokens_raw) == 0:
        trace['repair_trace_reason'] = 'no_repair_input_tokens'
        return trace

    pad_id = int(test_data.word2id.get('P', 0))
    start_id = int(test_data.word2id.get('S', 1))
    finish_id = int(test_data.word2id.get('F', 2))
    max_len = int(model.cfg.length_eq)
    id2word = test_data.id2word
    device = cfg_params.device

    try:
        true_tok_padded = tokenize_true_expression(
            true_expr=true_expr,
            word2id=test_data.word2id,
            max_len=max_len,
            pad_id=pad_id,
            finish_id=finish_id,
        )
        true_tok_trim = trim_token_sequence(true_tok_padded, pad_id=pad_id, finish_id=finish_id, include_finish=True)
        trace['repair_trace_true_token_ids'] = json_dumps_safe(true_tok_trim)
        trace['repair_trace_true_tokens'] = json_dumps_safe(token_ids_to_words(true_tok_trim, id2word))
    except Exception as e:
        trace['repair_trace_reason'] = f'true_expr_tokenization_failed: {e}'
        return trace

    # 与 fitfunc2 内部的 X / y 形状保持一致，用于 BFGS 与 train-MSE 早停判定
    X_model = X_tensor.clone().detach().to(device).unsqueeze(0)
    if X_model.shape[2] < model.cfg.dim_input - 1:
        pad = torch.zeros(1, X_model.shape[1], model.cfg.dim_input - X_model.shape[2] - 1, device=device)
        X_model = torch.cat((X_model, pad), dim=2)
    y_model = y_tensor[:, None].clone().detach().to(device).unsqueeze(0)

    X_cpu = X_model.cpu()
    y_cpu = y_model.cpu()
    repair_conf = float(getattr(cfg_params, 'repair_conf', 0.0))
    repair_stop_mse = float(getattr(cfg_params, 'repair_bfgs_stop_mse', 1.0e-4))
    n_vars = int(X_tensor.shape[1])
    eval_seed = int(getattr(cfg_params, 'eval_seed', 0))

    active_states = []
    for chain_rank, seed_tok in enumerate(seed_tokens_raw):
        init_tok = pack_token_sequence(
            seed_tok,
            pad_id=pad_id,
            start_id=start_id,
            finish_id=finish_id,
            max_len=max_len,
            device=device,
        )
        active_states.append({
            'tokens': init_tok,
            'score': 0.0,
            'raw_logp': 0.0,
            'trace': [],
            'chain_rank': int(chain_rank),
            'last_key': token_to_key(init_tok[0]),
        })

    all_records = {}
    best_record = None
    executed_iters = 0
    early_stop_iter = np.nan

    for step_idx in range(max_iters):
        step_candidates = []
        step_state_token_ids = []
        step_state_tokens = []
        step_neds = []
        next_active_states = []

        for state in active_states:
            step_out = model._repair_one_step_global(
                enc_out=src_enc,
                init_tokens=state['tokens'],
                conf_threshold=repair_conf,
                editor_beam_k=1,
                tagger_topk=1,
                n_vars=n_vars,
                trace_print=False,
            )
            if not step_out.get('successors'):
                continue

            succ = step_out['successors'][0]
            tok = succ['tokens'][0].detach().cpu()
            key = token_to_key(tok)
            if (not key) or key == state.get('last_key'):
                continue

            trimmed_ids = trim_token_sequence(tok, pad_id=pad_id, finish_id=finish_id, include_finish=True)
            step_state_token_ids.append(trimmed_ids)
            step_state_tokens.append(token_ids_to_words(trimmed_ids, id2word))
            step_neds.append(normalized_edit_distance(trimmed_ids, true_tok_trim))

            step_candidates.append({
                'token': tok,
                'score': float(state.get('raw_logp', 0.0) + succ.get('raw_logp', 0.0)),
                'raw_logp': float(state.get('raw_logp', 0.0) + succ.get('raw_logp', 0.0)),
                'trace': list(state.get('trace', [])) + (
                    [dict(succ['selected_edit'])] if succ.get('selected_edit', None) is not None else []
                ),
                'step_idx': int(step_idx + 1),
                'chain_rank': int(state.get('chain_rank', 0)),
                'last_key': key,
            })

        iter_no = step_idx + 1
        trace[f'repair_iter_{iter_no}_state_count'] = int(len(step_candidates))
        trace[f'repair_iter_{iter_no}_avg_norm_edit_distance'] = (
            float(np.mean(step_neds)) if len(step_neds) > 0 else np.nan
        )
        trace[f'repair_iter_{iter_no}_state_token_ids'] = json_dumps_safe(step_state_token_ids)
        trace[f'repair_iter_{iter_no}_state_tokens'] = json_dumps_safe(step_state_tokens)

        if len(step_candidates) == 0:
            break

        executed_iters = int(iter_no)
        eval_by_key = run_bfgs_on_tokens(
            token_list=[cand['token'] for cand in step_candidates],
            X_cpu=X_cpu,
            y_cpu=y_cpu,
            cfg_params=cfg_params,
            test_data=test_data,
            eval_seed=int(eval_seed + 20_000 + step_idx * 1_000),
        )

        for cand in step_candidates:
            key = cand.get('last_key')
            eval_info = eval_by_key.get(key)
            if eval_info is None:
                continue

            merged = dict(cand)
            merged.update(eval_info)
            merged['origin'] = f"repair_chain_{int(merged.get('chain_rank', 0))}"

            prev = all_records.get(key)
            if prev is None or better_record(prev, merged):
                all_records[key] = merged
            if better_record(best_record, merged):
                best_record = merged

            next_active_states.append({
                'tokens': merged['token'].to(device).unsqueeze(0),
                'score': float(merged.get('score', 0.0)),
                'raw_logp': float(merged.get('raw_logp', 0.0)),
                'trace': list(merged.get('trace', [])),
                'chain_rank': int(merged.get('chain_rank', 0)),
                'last_key': key,
            })

        if best_record is not None:
            best_mse_now = float(best_record.get('bfgs_loss', float('nan')))
            if np.isfinite(best_mse_now) and best_mse_now <= repair_stop_mse:
                early_stop_iter = int(iter_no)
                break

        active_states = next_active_states
        if len(active_states) == 0:
            break

    trace['repair_trace_executed_iters'] = int(executed_iters)
    trace['repair_trace_early_stop_hit'] = int(np.isfinite(early_stop_iter))
    trace['repair_trace_early_stop_iter'] = early_stop_iter
    if trace['repair_trace_reason'] == '':
        trace['repair_trace_reason'] = 'ok'
    return trace



def result_columns(max_repair_iters: int):
    cols = [
        'name', 'true_expr', 'predict_expr', 'test_r2', 'sr', 'base_time', 'repair_time', 'time', 'complexity',
        'ar_predict_expr', 'ar_test_r2', 'ar_sr', 'ar_complexity',
        'repair_predict_expr', 'repair_test_r2', 'repair_sr', 'repair_complexity', 'repair_best_origin',
        'repair_edit_steps', 'repair_search_executed_iters',
        'noise', 'train_range', 'test_range', 'repeat',
        'distractor_k', 'num_true_vars', 'total_vars', 'distractor_source_map',
        'ar_used_distractor', 'repair_used_distractor',
        'ar_num_used_distractors', 'repair_num_used_distractors',
        'repair_trace_source_beam', 'repair_trace_reason', 'repair_trace_skipped',
        'repair_trace_executed_iters', 'repair_trace_early_stop_hit', 'repair_trace_early_stop_iter',
        'repair_trace_true_token_ids', 'repair_trace_true_tokens',
    ]
    for i in range(1, max_repair_iters + 1):
        cols.extend([
            f'repair_iter_{i}_state_count',
            f'repair_iter_{i}_avg_norm_edit_distance',
            f'repair_iter_{i}_state_token_ids',
            f'repair_iter_{i}_state_tokens',
        ])
    return cols



def expr_to_func(sympy_expr, variables):
    import mpmath as mp2

    def cot(x):
        return 1 / np.tan(x)

    def acot(x):
        return 1 / np.arctan(x)

    def coth(x):
        return 1 / np.tanh(x)

    def harmonic_number(x):
        x_arr = np.asarray(x, dtype=np.float64)
        vec = np.vectorize(lambda t: float(mp2.digamma(t + 1.0) + mp2.euler), otypes=[float])
        return vec(x_arr)

    return lambdify(
        variables,
        sympy_expr,
        modules=["numpy", {"cot": cot, "acot": acot, "coth": coth, "harmonic": harmonic_number}],
    )



def compute_r2(y_gt, y_pred):
    return r2_score(y_gt, y_pred)



def evaluate_restored_prediction(raw_pred_str, scaler_x, scaler_y, variables, X_test_raw, y_test_raw):
    """
    将 BFGS 输出表达式恢复到原始量纲，并在测试集上计算 R2。
    返回: (expr_str, r2, complexity)
    """
    if raw_pred_str is None:
        return None, float("nan"), -1

    pre_expr = sp.sympify(raw_pred_str)
    expr_step1 = scaler_x.restore_x_expression(pre_expr)
    final_expr_obj = scaler_y.restore_y_expression(expr_step1)
    final_expr_obj = process_expr(sp.simplify(final_expr_obj))

    func_pred = lambdify(variables, final_expr_obj, modules="numpy")
    X_test_dict = {var: X_test_raw[:, idx] for idx, var in enumerate(variables)}
    y_pred_raw = func_pred(**X_test_dict)

    if isinstance(y_pred_raw, (float, int)):
        y_pred_raw = np.full_like(y_test_raw, y_pred_raw)
    elif np.ndim(y_pred_raw) > 1:
        y_pred_raw = np.asarray(y_pred_raw).flatten()
    if np.iscomplexobj(y_pred_raw):
        y_pred_raw = np.asarray(y_pred_raw).real

    r2 = r2_score(y_test_raw, y_pred_raw)
    expr_str = str(final_expr_obj)
    complexity = calculate_tree_size(expr_str)
    return expr_str, float(r2), int(complexity)



def evaluate_points(func, points):
    vals = func(*[points[:, i] for i in range(points.shape[1])])
    if np.ndim(vals) == 0:
        vals = np.full(points.shape[0], vals)
    y = np.reshape(vals, (-1, 1))
    if y.shape[0] != points.shape[0]:
        y = np.broadcast_to(y, (points.shape[0], 1))
    if np.iscomplexobj(y) or np.any(np.iscomplex(y)):
        return np.full((points.shape[0], 1), np.nan)
    return y.astype(np.float64)



def get_variable_names(expr: str):
    variables = re.findall(r'x_\d+', expr)
    unique_vars = sorted(set(variables), key=lambda x: int(x.split('_')[1]))
    return unique_vars



def pad_to_10_columns(tensor):
    n = tensor.size(1)
    pad_cols = max(0, 10 - n)
    padded_tensor = F.pad(tensor, (0, pad_cols), mode='constant', value=0)
    return padded_tensor



def parse_range_spec(range_str: str, num_vars: int):
    text = str(range_str).strip()
    if ":" not in text:
        m = _UE_PAT.match(text)
        if not m:
            raise ValueError(f"无法解析范围: {range_str}")
        mode, lo, hi, n = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))
        return {i: (mode, lo, hi, n) for i in range(1, num_vars + 1)}

    specs = {}
    parts = [p.strip() for p in text.split(";") if p.strip()]
    for p in parts:
        left, right = p.split(":", 1)
        left = left.strip()
        right = right.strip()

        vars_ = []
        for token in left.split(","):
            token = token.strip()
            mvar = re.match(r"^x_(\d+)$", token)
            if not mvar:
                raise ValueError(f"无法解析变量名: {token}  (完整: {range_str})")
            vars_.append(int(mvar.group(1)))

        m = _UE_PAT.match(right)
        if not m:
            raise ValueError(f"无法解析 U/E 范围: {right}  (完整: {range_str})")
        mode, lo, hi, n = m.group(1), float(m.group(2)), float(m.group(3)), int(m.group(4))

        for vidx in vars_:
            specs[vidx] = (mode, lo, hi, n)

    missing = [i for i in range(1, num_vars + 1) if i not in specs]
    if missing:
        raise ValueError(f"范围未覆盖全部变量: missing={missing}  (完整: {range_str})")
    return specs



def sample_X(specs, num_vars: int, seed: int) -> np.ndarray:
    n_set = {specs[i][3] for i in range(1, num_vars + 1)}
    if len(n_set) != 1:
        raise ValueError(f"不同变量的采样点数不一致: {n_set}")
    n = next(iter(n_set))

    rng = np.random.default_rng(seed)
    cols = []
    for i in range(1, num_vars + 1):
        mode, lo, hi, n_i = specs[i]
        if mode == "U":
            col = rng.uniform(lo, hi, size=n_i)
        elif mode == "E":
            col = np.linspace(lo, hi, num=n_i, dtype=np.float64)
            if num_vars > 1:
                rng.shuffle(col)
        else:
            raise ValueError(f"未知采样模式: {mode}")
        cols.append(col.astype(np.float64))

    X = np.column_stack(cols)
    if X.shape[0] != n:
        raise ValueError(f"采样点数异常: 期望 {n}, 实际 {X.shape[0]}")
    return X



def sample_points_from_specs(func, num_vars: int, specs, target_noise: float, seed: int):
    X = sample_X(specs, num_vars=num_vars, seed=seed)
    y_truth = np.squeeze(evaluate_points(func, X))
    is_valid = np.isfinite(y_truth)
    X = X[is_valid]
    y_truth = y_truth[is_valid]

    if len(y_truth) < 10:
        raise ValueError("有效样本过少 (NaN/Inf 过多)")

    if target_noise > 0:
        scale = target_noise * np.sqrt(np.mean(np.square(y_truth)))
        if scale == 0:
            noise = np.zeros_like(y_truth)
        else:
            # 与 rils-rols_std.py 保持一致：噪声生成器固定种子。
            rng = np.random.default_rng(42)
            noise = rng.normal(loc=0.0, scale=scale, size=y_truth.shape)
        y_noisy = y_truth + noise
    else:
        y_noisy = y_truth

    return X, y_noisy, y_truth


def sample_one_column_from_spec(spec, seed: int):
    mode, lo, hi, n = spec
    rng = np.random.default_rng(seed)
    if mode == "U":
        col = rng.uniform(lo, hi, size=n)
    elif mode == "E":
        col = np.linspace(lo, hi, num=n, dtype=np.float64)
        rng.shuffle(col)
    else:
        raise ValueError(f"未知采样模式: {mode}")
    return np.asarray(col, dtype=np.float64)


def build_random_distractor_specs(train_specs, test_specs, num_true_vars: int, num_distractors: int, seed: int):
    """
    为新增干扰变量随机复用已有变量的采样范围。
    返回:
      extra_train_specs: {new_var_idx: spec}
      extra_test_specs:  {new_var_idx: spec}
      source_map:        {new_var_idx: copied_from_true_var_idx}
    """
    rng = np.random.default_rng(seed)
    extra_train_specs = {}
    extra_test_specs = {}
    source_map = {}
    for j in range(num_distractors):
        new_vidx = num_true_vars + j + 1
        src_vidx = int(rng.integers(1, num_true_vars + 1))
        extra_train_specs[new_vidx] = train_specs[src_vidx]
        extra_test_specs[new_vidx] = test_specs[src_vidx]
        source_map[new_vidx] = src_vidx
    return extra_train_specs, extra_test_specs, source_map


def append_random_distractors_to_train_test(
    X_train_raw,
    X_test_raw,
    train_specs,
    test_specs,
    num_true_vars: int,
    num_distractors: int,
    seed: int,
):
    if num_distractors <= 0:
        all_variables = [f"x_{i}" for i in range(1, num_true_vars + 1)]
        return X_train_raw, X_test_raw, all_variables, {}

    extra_train_specs, extra_test_specs, source_map = build_random_distractor_specs(
        train_specs=train_specs,
        test_specs=test_specs,
        num_true_vars=num_true_vars,
        num_distractors=num_distractors,
        seed=seed,
    )

    extra_train_cols = []
    extra_test_cols = []
    for offset, new_vidx in enumerate(range(num_true_vars + 1, num_true_vars + num_distractors + 1)):
        extra_train_cols.append(
            sample_one_column_from_spec(extra_train_specs[new_vidx], seed=seed + 1000 + offset)
        )
        extra_test_cols.append(
            sample_one_column_from_spec(extra_test_specs[new_vidx], seed=seed + 2000 + offset)
        )

    X_train_aug = np.column_stack([X_train_raw] + extra_train_cols)
    X_test_aug = np.column_stack([X_test_raw] + extra_test_cols)
    all_variables = [f"x_{i}" for i in range(1, num_true_vars + num_distractors + 1)]
    return X_train_aug, X_test_aug, all_variables, source_map


def get_used_var_indices(expr_str):
    if expr_str is None:
        return set()
    vars_found = re.findall(r'x_(\d+)', str(expr_str))
    return {int(v) for v in vars_found}


def count_used_distractors(expr_str, num_true_vars: int):
    used = get_used_var_indices(expr_str)
    return sum(1 for vidx in used if vidx > num_true_vars)


def uses_any_distractor(expr_str, num_true_vars: int):
    return int(count_used_distractors(expr_str, num_true_vars) > 0)


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
    """不执行任何缩放的占位符类"""

    def fit(self, X):
        pass

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


# ==========================================
# 主程序
# ==========================================

@hydra.main(config_name="config", version_base='1.1')
def main(cfg):
    SCALER_TYPE = 'none'  # 可选: 'auto', 'zscore', 'minmax', 'none'
    REPEATS = 3
    TRAIN_NOISE = 0.0
    BASE_SEED = 0
    BEAMS = [10, 20, 30]
    BEAM50_FORCE_AUTO_SCALER = True
    MAX_REPAIR_ITERS = int(getattr(cfg.inference, 'repair_max_iters', 10))
    MAX_DISTRACTOR_K = 3
    DISTRACTOR_MIN_K = 3

    print(f"=== 使用归一化策略: {SCALER_TYPE} ===")
    print("=== 使用 table3_with_n200.csv 进行测试（随机 distractor stress-test） ===")
    print(f"=== 每个任务重复次数: {REPEATS} ===")
    print(f"=== 每个任务随机添加的干扰变量数: [{DISTRACTOR_MIN_K}, {MAX_DISTRACTOR_K}]，且不超过 10 维输入上限 ===")

    metadata_path = resolve_first_existing_path([
        str(scripts_path("data", "val_data", "10vars", "100")),
        str(resolve_path("data/val_data/10vars/100", base="scripts")),
        str(resolve_path("scripts/data/val_data/10vars/100", base="project")),
    ])
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

    model_path = resolve_path(cfg.model_path, base="scripts")
    model = Model.load_from_checkpoint(model_path, cfg=cfg.architecture)
    model.eval().to(cfg.inference.device)
    fitfunc = partial(model.fitfunc2, cfg_params=params_fit)

    encoder_tap, original_encoder_forward = install_encoder_tap(model)

    benchmarks_path = resolve_first_existing_path([
        str(resolve_path("table3_with_n200_1.csv", base="scripts")),
        str(resolve_path("scripts/table3_with_n200_1.csv", base="project")),
        str(resolve_path("data/table3_with_n200_1.csv", base="scripts")),
        str(scripts_path("table3_with_n200_1.csv")),
    ])
    benchmarks = pd.read_csv(benchmarks_path)

    required = {"name", "variables", "expression", "train_range", "test_range"}
    missing_cols = required - set(benchmarks.columns)
    if missing_cols:
        raise ValueError(f"CSV 缺少列: {missing_cols}, 当前列为: {list(benchmarks.columns)}")

    results = []
    all_columns = result_columns(MAX_REPAIR_ITERS)

    try:
        for row_idx, row in benchmarks.iterrows():
            name = str(row["name"])
            print("#####################################################################################")
            print(f"Task: {name}")

            for rep in range(REPEATS):
                print("**********************************************************")
                print(f"Repeat: {rep + 1}/{REPEATS}")

                try:
                    noise = float(TRAIN_NOISE)
                    num_var = int(row["variables"])
                    raw_expr = str(row["expression"])
                    train_range_str = str(row["train_range"])
                    test_range_str = str(row["test_range"])

                    expr = process_benchmark_expression(raw_expr)
                    true_variables = [f"x_{i}" for i in range(1, num_var + 1)]
                    print("Expr:", expr)
                    print("True Vars:", true_variables)
                    print("Train range:", train_range_str)
                    print("Test range:", test_range_str)

                    sym_eq = sympify(expr)
                    lam = expr_to_func(sym_eq, true_variables)

                    train_specs = parse_range_spec(train_range_str, num_var)
                    test_specs = parse_range_spec(test_range_str, num_var)

                    seed_train = BASE_SEED + row_idx * 100 + rep * 10 + 1
                    seed_test = BASE_SEED + row_idx * 100 + rep * 10 + 2
                    seed_distractor = BASE_SEED + row_idx * 100 + rep * 10 + 3

                    X_raw_base, y_raw, _ = sample_points_from_specs(
                        lam, num_vars=num_var, specs=train_specs, target_noise=noise, seed=seed_train
                    )
                    X_test_raw_base, _, y_test_raw = sample_points_from_specs(
                        lam, num_vars=num_var, specs=test_specs, target_noise=0.0, seed=seed_test
                    )

                    feasible_max_k = min(int(MAX_DISTRACTOR_K), max(0, 10 - num_var))
                    if feasible_max_k < DISTRACTOR_MIN_K:
                        print(f"Skip task={name}: 原始变量数 {num_var} 已接近/达到 10 维，无法添加 distractor。")
                        continue

                    rng_k = np.random.default_rng(seed_distractor)
                    distractor_k = int(rng_k.integers(DISTRACTOR_MIN_K, feasible_max_k + 1))
                    X_raw, X_test_raw, variables, distractor_source_map = append_random_distractors_to_train_test(
                        X_train_raw=X_raw_base,
                        X_test_raw=X_test_raw_base,
                        train_specs=train_specs,
                        test_specs=test_specs,
                        num_true_vars=num_var,
                        num_distractors=distractor_k,
                        seed=seed_distractor,
                    )

                    print(f"Distractor k: {distractor_k} | Total Vars: {len(variables)}")
                    print(f"All Vars: {variables}")
                    print(f"Distractor source map: {distractor_source_map}")

                    cfg_params = OmegaConf.create(OmegaConf.to_container(cfg.inference, resolve=True))
                    cfg_params.use_repair = True
                    cfg_params.return_baseline_and_repair = True

                    ar_best_expr = None
                    ar_best_train_r2 = -np.inf
                    ar_best_r2 = -np.inf  # test R2 of the AR candidate selected by train R2
                    ar_best_complexity = -1

                    repair_best_expr = None
                    repair_best_train_r2 = -np.inf
                    repair_best_r2 = -np.inf  # test R2 of the repair candidate selected by train R2
                    repair_best_complexity = -1
                    repair_best_origin = None
                    repair_best_time = float('nan')
                    repair_best_edit_steps = 0
                    repair_best_search_executed_iters = 0
                    repair_trace_best = empty_repair_trace_dict(MAX_REPAIR_ITERS)
                    repair_trace_initialized = False

                    ar_best_time = float('nan')
                    main_time = float('nan')

                    for beam in BEAMS:
                        try:
                            current_scaler_type = 'auto' if (beam == 100 and BEAM50_FORCE_AUTO_SCALER) else SCALER_TYPE

                            if current_scaler_type == 'zscore':
                                scaler_x = ZScoreScaler()
                                scaler_y = ZScoreScaler()
                                scaler_x.fit(X_raw)
                                scaler_y.fit(y_raw)
                            elif current_scaler_type == 'minmax':
                                scaler_x = MinMaxScaler()
                                scaler_y = MinMaxScaler()
                                scaler_x.fit(X_raw)
                                scaler_y.fit(y_raw)
                            elif current_scaler_type == 'none':
                                scaler_x = IdentityScaler()
                                scaler_y = IdentityScaler()
                            elif current_scaler_type == 'auto':
                                scaler_x = AutoMagnitudeScaler().fit(X=X_raw)
                                scaler_y = AutoMagnitudeScaler().fit(y_raw)
                            else:
                                raise ValueError(f"未知 SCALER_TYPE: {current_scaler_type}")

                            if current_scaler_type == 'auto':
                                X_scaled = scaler_x.transform(X=X_raw)
                                y_scaled = scaler_y.transform(y_raw).ravel()
                            else:
                                X_scaled = scaler_x.transform(X_raw)
                                y_scaled = scaler_y.transform(y_raw).ravel()

                            X_tensor = pad_to_10_columns(torch.tensor(X_scaled).float())
                            y_tensor = torch.tensor(y_scaled).float()

                            print(f"Beam {beam} | scaler = {current_scaler_type}")

                            cfg_params.beam_size = beam
                            cfg_params.repair_seed_topk = beam
                            output = fitfunc(X_tensor, y_tensor, cfg_params=cfg_params, test_data=test_data)

                            beam_base_time = float(output.get('baseline_elapsed', float('nan')))
                            beam_repair_extra_time = float(output.get('repair_elapsed', float('nan')))
                            beam_total_time = float(output.get('total_elapsed', float('nan')))

                            src_enc_for_this_beam = encoder_tap.get('last_output', None)

                            ar_raw_pred = None
                            if output.get('baseline_best_bfgs_preds') is not None:
                                ar_raw_pred = output['baseline_best_bfgs_preds'][0]

                            repair_raw_pred = None
                            if output.get('repair_best_bfgs_preds') is not None:
                                repair_raw_pred = output['repair_best_bfgs_preds'][0]

                            if ar_raw_pred is not None:
                                ar_expr_str, ar_train_r2, ar_complexity = evaluate_restored_prediction(
                                    ar_raw_pred, scaler_x, scaler_y, variables, X_raw, y_raw
                                )
                                _, ar_r2, _ = evaluate_restored_prediction(
                                    ar_raw_pred, scaler_x, scaler_y, variables, X_test_raw, y_test_raw
                                )
                                print(f"Beam {beam} | AR     Train/Test R2: {ar_train_r2:.4f}/{ar_r2:.4f} | Expr: {ar_expr_str}")
                                if np.isfinite(ar_train_r2) and ar_train_r2 > ar_best_train_r2:
                                    ar_best_train_r2 = float(ar_train_r2)
                                    ar_best_r2 = float(ar_r2)
                                    ar_best_expr = ar_expr_str
                                    ar_best_complexity = int(ar_complexity)
                                    ar_best_time = beam_base_time

                            if repair_raw_pred is not None:
                                repair_expr_str, repair_train_r2, repair_complexity = evaluate_restored_prediction(
                                    repair_raw_pred, scaler_x, scaler_y, variables, X_raw, y_raw
                                )
                                _, repair_r2, _ = evaluate_restored_prediction(
                                    repair_raw_pred, scaler_x, scaler_y, variables, X_test_raw, y_test_raw
                                )
                                print(f"Beam {beam} | Repair Train/Test R2: {repair_train_r2:.4f}/{repair_r2:.4f} | Expr: {repair_expr_str}")

                                need_collect_trace = (not repair_trace_initialized)
                                if np.isfinite(repair_train_r2) and repair_train_r2 > repair_best_train_r2:
                                    need_collect_trace = True

                                current_trace = None
                                if need_collect_trace:
                                    try:
                                        current_trace = collect_repair_trace_stats(
                                            model=model,
                                            output=output,
                                            src_enc=src_enc_for_this_beam,
                                            X_tensor=X_tensor,
                                            y_tensor=y_tensor,
                                            cfg_params=cfg_params,
                                            test_data=test_data,
                                            true_expr=str(sym_eq),
                                            source_beam=beam,
                                        )
                                        repair_trace_initialized = True
                                    except Exception as trace_e:
                                        current_trace = empty_repair_trace_dict(MAX_REPAIR_ITERS)
                                        current_trace['repair_trace_source_beam'] = int(beam)
                                        current_trace['repair_trace_reason'] = f'trace_collection_failed: {trace_e}'
                                        repair_trace_initialized = True

                                if np.isfinite(repair_train_r2) and repair_train_r2 > repair_best_train_r2:
                                    repair_best_train_r2 = float(repair_train_r2)
                                    repair_best_r2 = float(repair_r2)
                                    repair_best_expr = repair_expr_str
                                    repair_best_complexity = int(repair_complexity)
                                    repair_best_origin = output.get('repair_best_origin')
                                    repair_best_time = beam_total_time
                                    repair_best_edit_steps = int(output.get('repair_best_edit_steps', 0) or 0)
                                    repair_best_search_executed_iters = int(output.get('repair_search_executed_iters', 0) or 0)
                                    if current_trace is not None:
                                        repair_trace_best = current_trace
                                elif (current_trace is not None) and (not np.isfinite(repair_best_r2)):
                                    repair_trace_best = current_trace

                            # 只用训练集 R2 触发早停；测试集只用于最终报告。
                            ar_hit = np.isfinite(ar_best_train_r2) and (ar_best_train_r2 >= 0.999)
                            repair_hit = np.isfinite(repair_best_train_r2) and (repair_best_train_r2 >= 0.999)
                            if ar_hit or repair_hit:
                                print(
                                    f"Early stop at beam={beam} | "
                                    f"ar_best_train_r2={ar_best_train_r2:.4f}, repair_best_train_r2={repair_best_train_r2:.4f}"
                                )
                                break

                        except Exception as e:
                            print(f"Beam {beam} Error: {e}")
                            continue

                    if ar_best_expr is None:
                        ar_best_train_r2 = float('nan')
                        ar_best_r2 = float('nan')
                    ar_sr = int(symbol_equivalence_single(str(sym_eq), ar_best_expr, variables)) if ar_best_expr is not None else 0

                    if repair_best_expr is None:
                        repair_best_train_r2 = float('nan')
                        repair_best_r2 = float('nan')
                    repair_sr = int(symbol_equivalence_single(str(sym_eq), repair_best_expr, variables)) if repair_best_expr is not None else 0

                    # 为兼容旧列名，默认将主结果指向 repair；若 repair 不存在，则回退到 AR。
                    main_predict_expr = repair_best_expr if repair_best_expr is not None else ar_best_expr
                    main_selection_train_r2 = repair_best_train_r2 if repair_best_expr is not None else ar_best_train_r2
                    main_test_r2 = repair_best_r2 if repair_best_expr is not None else ar_best_r2
                    main_sr = repair_sr if repair_best_expr is not None else ar_sr
                    main_complexity = repair_best_complexity if repair_best_expr is not None else ar_best_complexity

                    if np.isfinite(main_test_r2) and (repair_best_expr is not None) and (main_predict_expr == repair_best_expr):
                        main_time = repair_best_time
                    elif np.isfinite(main_test_r2) and (ar_best_expr is not None) and (main_predict_expr == ar_best_expr):
                        main_time = ar_best_time

                    row_result = {
                        'name': name,
                        'true_expr': str(sym_eq),
                        'predict_expr': main_predict_expr,
                        'selection_train_r2': main_selection_train_r2,
                        'test_r2': main_test_r2,
                        'sr': int(main_sr),
                        'base_time': ar_best_time,
                        'repair_time': repair_best_time,
                        'time': main_time,
                        'complexity': main_complexity,
                        'ar_predict_expr': ar_best_expr,
                        'ar_selection_train_r2': ar_best_train_r2,
                        'ar_test_r2': ar_best_r2,
                        'ar_sr': int(ar_sr),
                        'ar_complexity': ar_best_complexity,
                        'repair_predict_expr': repair_best_expr,
                        'repair_selection_train_r2': repair_best_train_r2,
                        'repair_test_r2': repair_best_r2,
                        'repair_sr': int(repair_sr),
                        'repair_complexity': repair_best_complexity,
                        'repair_best_origin': repair_best_origin,
                        'repair_edit_steps': int(repair_best_edit_steps),
                        'repair_search_executed_iters': int(repair_best_search_executed_iters),
                        'noise': noise,
                        'train_range': train_range_str,
                        'test_range': test_range_str,
                        'repeat': rep,
                        'distractor_k': int(distractor_k),
                        'num_true_vars': int(num_var),
                        'total_vars': int(len(variables)),
                        'distractor_source_map': json.dumps(distractor_source_map, ensure_ascii=False),
                        'ar_used_distractor': uses_any_distractor(ar_best_expr, num_var),
                        'repair_used_distractor': uses_any_distractor(repair_best_expr, num_var),
                        'ar_num_used_distractors': count_used_distractors(ar_best_expr, num_var),
                        'repair_num_used_distractors': count_used_distractors(repair_best_expr, num_var),
                    }
                    row_result.update(repair_trace_best)

                    for col in all_columns:
                        row_result.setdefault(col, np.nan if 'distance' in col or 'iter' in col or col.endswith('_beam') else '')

                    results.append(row_result)

                    print(
                        f"[Result] {name} | repeat={rep + 1} | Train/Test R2={main_selection_train_r2:.4f}/{main_test_r2:.4f} | "
                        f"AR={ar_best_train_r2:.4f}/{ar_best_r2:.4f} | Repair={repair_best_train_r2:.4f}/{repair_best_r2:.4f} | "
                        f"SR={int(main_sr)} | DistractorK={int(distractor_k)} | "
                        f"AR_used={int(bool(uses_any_distractor(ar_best_expr, num_var)))} | "
                        f"Repair_used={int(bool(uses_any_distractor(repair_best_expr, num_var)))}"
                    )

                    del X_raw_base, X_raw, y_raw, X_test_raw_base, X_test_raw, y_test_raw, X_tensor, y_tensor
                    gc.collect()

                except Exception as e:
                    print(f"发生异常: task={name}, repeat={rep + 1}, error={e}")
                    import traceback
                    traceback.print_exc()
                    continue
    finally:
        uninstall_encoder_tap(model, original_encoder_forward)


if __name__ == "__main__":
    main()
