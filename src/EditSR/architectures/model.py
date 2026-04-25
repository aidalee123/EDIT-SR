import torch
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
try:
    from .MultimodalEncoder import MultiModalEncoder
    from .diffusion_repair import PrefixRepairHelper, constrained_decode_batch_from_position_logits, \
        constrained_decode_body_from_slot_logits
except ImportError:
    from MultimodalEncoder import MultiModalEncoder
    from diffusion_repair import PrefixRepairHelper, constrained_decode_batch_from_position_logits, \
        constrained_decode_body_from_slot_logits

# Repair actions (root-level)
ACT_KEEP = 0
ACT_REPLACE = 1
ACT_OP_REPLACE = ACT_REPLACE  # backward-compat alias
ACT_DELETE_SUBTREE = 2
ACT_REWRITE_SUBTREE = 3
ACT_INSERT = 4
NUM_REPAIR_ACTIONS = 5
TAGGER_ACTIONS = (ACT_KEEP, ACT_REPLACE, ACT_DELETE_SUBTREE, ACT_REWRITE_SUBTREE, ACT_INSERT)
NUM_TAGGER_ACTIONS = len(TAGGER_ACTIONS)

try:
    from .beam_search import BeamHypotheses
except ImportError:
    from beam_search import BeamHypotheses
import numpy as np
try:
    from . import bfgs
except ImportError:
    import bfgs
import math
import random
import sympy as sp
from typing import Any, Dict, List, Optional, Set, Tuple
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import r2_score


def bfgs_wrapper(args):
    """Run BFGS in a subprocess.

    Args tuple:
      (ww, X_cpu, y_cpu, cfg_params, test_data, seed)
    where seed can be None.
    """
    if len(args) == 5:
        ww, X_cpu, y_cpu, cfg_params, test_data = args
        seed = None
    else:
        ww, X_cpu, y_cpu, cfg_params, test_data, seed = args

    # Deterministic BFGS restarts (important for clean ablations)
    if seed is not None:
        try:
            np.random.seed(int(seed) % (2 ** 32 - 1))
            random.seed(int(seed) % (2 ** 32 - 1))
        except Exception:
            pass

    try:
        pred_w_c, constants, loss_bfgs, exa = bfgs.bfgs(ww, X_cpu, y_cpu, cfg_params, test_data)
        return (str(pred_w_c), loss_bfgs, ww)
    except Exception:
        return (None, float("nan"), ww)


# =============================================================================
# Rooted edit-chain repair helper
# =============================================================================


class Model(pl.LightningModule):
    def _metric_name_allowed_for_logger(self, name: str) -> bool:
        key = str(name).lower()
        return ('loss' in key) or ('acc' in key)

    def log(self, name, value, *args, **kwargs):
        if kwargs.get('logger', True) and not self._metric_name_allowed_for_logger(str(name)):
            kwargs['logger'] = False
        return super().log(name, value, *args, **kwargs)

    def log_dict(self, dictionary, *args, **kwargs):
        if kwargs.get('logger', True):
            filtered = {k: v for k, v in dictionary.items() if self._metric_name_allowed_for_logger(str(k))}
            if not filtered:
                return None
            dictionary = filtered
        return super().log_dict(dictionary, *args, **kwargs)

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.automatic_optimization = False

        # Core Architecture
        self.MultiModalEncoder = MultiModalEncoder(cfg)

        # Decoder (Symbolic Generation)
        self.trg_pad_idx = cfg.trg_pad_idx
        self.tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
        self.pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
        if bool(getattr(cfg, "sinuisodal_embeddings", False)):
            self.create_sinusoidal_embeddings(cfg.length_eq, cfg.dim_hidden, out=self.pos_embedding.weight)
        # 预先缓存位置索引，减少每次 forward 里重复构造 arange 的开销
        self.register_buffer("_pos_idx_eq", torch.arange(cfg.length_eq, dtype=torch.long), persistent=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.dim_hidden,
            nhead=cfg.num_heads,
            dim_feedforward=int(getattr(cfg, "dec_pf_dim", cfg.dim_hidden)),
            dropout=float(getattr(cfg, "dropout", 0.0)),
        )
        self.decoder_transfomer = nn.TransformerDecoder(decoder_layer, num_layers=cfg.dec_layers)
        self.fc_out = nn.Linear(cfg.dim_hidden, cfg.output_dim)
        self.dropout = nn.Dropout(float(getattr(cfg, "dropout", 0.0)))

        # Loss (symbolic token generation)
        self.CrossEntropy_Loss = nn.CrossEntropyLoss(ignore_index=int(cfg.trg_pad_idx))

        self.num_epochs = cfg.epochs

        # ---------------- Vocabulary cache (filled from data.py metadata at eval time) ----------------
        # NOTE: vocab is provided by EditSRDataset metadata at eval time; do NOT read it from cfg.
        self.word2id: Optional[Dict[str, int]] = None
        self.id2word: Optional[Dict[int, str]] = None
        self._vocab_ready: bool = False
        self._repair_helper: Optional[PrefixRepairHelper] = None

        # ---------------- Repair Editor config (must be specified in config.yaml -> architecture.*) ----------------
        # Whether to use explicit sentinel/tag tokens (e.g. <repl:0>/<insert:0>) to mark global edit skeleton spans.
        self.repair_gen_use_tag_tokens: bool = bool(cfg.repair_editor_use_tag_tokens)
        # Whether to also add action embedding (may be redundant when using tag tokens).
        self.repair_gen_use_action_embedding: bool = bool(cfg.repair_editor_use_action_embedding)
        # Whether to use infill/prefix-LM self-attention mask in the Editor (holes see both left+right context, while each hole block stays AR).
        self.repair_gen_use_infill_mask: bool = bool(cfg.repair_editor_use_infill_mask)
        # Tag token strings (will be injected into vocab on_fit_start).
        self.repair_tag_repl_str: str = str(cfg.repair_tag_repl_str)
        self.repair_tag_insert_str: str = str(cfg.repair_tag_insert_str)
        # Resolved ids after vocab is cached
        self.repair_tag_repl_id: Optional[int] = None
        self.repair_tag_insert_id: Optional[int] = None

        # Token ids that must NEVER be generated (e.g. explicit repair tag tokens like <repl:0>/<insert:0>).
        self._forbidden_output_ids: Optional[List[int]] = None
        # Cache causal masks by (device, length) to avoid rebuilding [L,L] masks every decode step.
        self._causal_mask_cache: Dict[Tuple[str, int], torch.Tensor] = {}

        # ---------------- Rooted edit-chain repair config (must be specified in config.yaml -> architecture.*) ----------------
        self.repair_enable: bool = bool(cfg.repair_enable)
        self.repair_train: bool = bool(cfg.repair_train)
        self.repair_only: bool = bool(cfg.repair_only)
        self.repair_only_freeze_ar: bool = bool(cfg.repair_only_freeze_ar)

        self.repair_layers: int = int(cfg.repair_layers)

        # Training-state sources.
        self.repair_source_use_ar: bool = bool(cfg.repair_source_use_ar)
        self.repair_source_use_synth: bool = bool(cfg.repair_source_use_synth)
        self.repair_source_use_rollout: bool = bool(cfg.repair_source_use_rollout)
        self.repair_source_rollout_prob: float = float(cfg.repair_source_rollout_prob)
        self.repair_source_rollout_steps: int = int(max(1, int(cfg.repair_source_rollout_steps)))

        # Source-chain construction and chain-state sampling.
        self.repair_chain_depth_min: int = int(max(1, int(cfg.repair_chain_depth_min)))
        self.repair_chain_depth_max: int = int(max(self.repair_chain_depth_min, int(cfg.repair_chain_depth_max)))
        self.repair_chain_resample_attempts: int = int(max(1, int(cfg.repair_chain_resample_attempts)))
        self.repair_chain_state_sampling: str = str(cfg.repair_chain_state_sampling).lower()
        self.repair_supervision_mode: str = str(getattr(cfg, 'repair_supervision_mode', 'full_chain_single_edit')).lower()

        # Inference-time maximum number of repair iterations (search budget H).
        self.repair_conf_threshold: float = float(cfg.repair_conf_threshold)
        self.repair_tagger_keep_weight: float = float(getattr(cfg, 'repair_tagger_keep_weight', 0.5))
        self.repair_tagger_keep_self_prob: float = float(min(1.0, max(0.0, float(getattr(cfg, 'repair_tagger_keep_self_prob', 0.15)))))
        self.repair_tagger_use_action_mask: bool = bool(getattr(cfg, 'repair_tagger_use_action_mask', True))
        self.repair_tagger_class_balance_power: float = float(max(0.0, float(getattr(cfg, 'repair_tagger_class_balance_power', 0.5))))
        self.repair_tagger_min_class_prop: float = float(max(1e-6, float(getattr(cfg, 'repair_tagger_min_class_prop', 0.01))))
        self.repair_tagger_max_class_weight: float = float(max(1.0, float(getattr(cfg, 'repair_tagger_max_class_weight', 4.0))))
        self.repair_tagger_replace_weight: float = float(max(0.0, float(getattr(cfg, 'repair_tagger_replace_weight', 1.5))))
        self.repair_tagger_delete_weight: float = float(max(0.0, float(getattr(cfg, 'repair_tagger_delete_weight', 2.0))))
        self.repair_tagger_rewrite_weight: float = float(max(0.0, float(getattr(cfg, 'repair_tagger_rewrite_weight', 1.5))))
        self.repair_tagger_insert_weight: float = float(max(0.0, float(getattr(cfg, 'repair_tagger_insert_weight', 1.5))))

        self.repair_corruption_weight_replace: float = float(max(0.0, float(getattr(cfg, 'repair_corruption_weight_replace', 1.5))))
        self.repair_corruption_weight_delete: float = float(max(0.0, float(getattr(cfg, 'repair_corruption_weight_delete', 2.0))))
        self.repair_corruption_weight_rewrite: float = float(max(0.0, float(getattr(cfg, 'repair_corruption_weight_rewrite', 1.5))))
        self.repair_corruption_weight_insert: float = float(max(0.0, float(getattr(cfg, 'repair_corruption_weight_insert', 1.5))))

        # Debug-only overfit cache and CSV tracing are disabled in the normal training configuration.
        self.repair_overfit_debug_enable: bool = False
        self.repair_overfit_debug_train_batches: int = 1
        self.repair_overfit_debug_val_batches: int = 1
        self.repair_overfit_debug_use_train_cache_for_val: bool = False
        self._repair_overfit_train_cache: List[Any] = []
        self._repair_overfit_val_cache: List[Any] = []

        self.repair_trace_enable: bool = False
        self.repair_trace_train_only: bool = True
        self.repair_trace_cases_per_batch: int = 0
        self.repair_trace_max_cases: int = 0
        self.repair_trace_output_csv: str = 'repair_supervision_trace.csv'
        self._repair_trace_cases_written: int = 0
        self._repair_trace_rows_written: int = 0
        self._repair_trace_header_written: bool = False
        self._repair_trace_csv_path: Optional[str] = None
        self._repair_trace_fieldnames: List[str] = [
            'trace_id', 'split', 'epoch', 'global_step', 'batch_idx', 'source_batch_idx', 'source_type', 'row_type',
            'row_index', 'distribution_name', 'sampled_value', 'chain_length', 'frontier_count',
            'gt_body_ids', 'gt_body_tokens', 'source_body_ids', 'source_body_tokens',
            'state_body_ids', 'state_body_tokens', 'before_body_ids', 'before_body_tokens',
            'after_body_ids', 'after_body_tokens', 'action_name', 'action_id', 'root_idx', 'path',
            'span_start', 'span_end', 'prev_span', 'cur_span', 'target_token_id', 'target_token_str',
            'target_subtree_ids', 'target_subtree_tokens', 'edited_subtree_ids', 'edited_subtree_tokens',
            'frontier_labels_json', 'extra_json'
        ]

        # Frontier supervision / oracle parameters.
        self.repair_frontier_lambda: float = float(cfg.repair_frontier_lambda)
        self.repair_direct_rewrite_max_nodes: int = int(max(2, int(cfg.repair_direct_rewrite_max_nodes)))

        # Values that materially affect legality masking / decoding must come from config.
        self.forbidden_logit_value: float = float(cfg.forbidden_logit_value)

        # Try initialize helper from cached vocab if present (will also be refreshed on_fit_start)
        self._maybe_init_vocab_helper()

        # Keep repair decoders configurable with the *same style* as the NeSymReS main decoder.
        # By default they mirror dec_pf_dim / dropout / sinusoidal setting, but can be overridden
        # independently from config.yaml if desired.
        self.repair_dec_pf_dim: int = int(cfg.repair_dec_pf_dim)
        self.repair_dropout_p: float = float(cfg.repair_dropout)
        self.repair_use_sinusoidal_embeddings: bool = bool(
            cfg.repair_sinuisodal_embeddings
        )
        if self.repair_only and self.repair_enable and self.repair_train and self.repair_only_freeze_ar:
            self.tok_embedding.requires_grad_(False)
            self.decoder_transfomer.requires_grad_(False)
            self.fc_out.requires_grad_(False)

        if self.repair_enable:
            # Keep the numeric condition encoder shared with AR, but detach its output inside the
            # repair branch so repair gradients do not flow back into MultiModalEncoder.

            # Tagger: bidirectional conditional Transformer over the current packed expression.
            self.repair_tagger_tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
            self.repair_tagger_pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
            if self.repair_use_sinusoidal_embeddings:
                self.create_sinusoidal_embeddings(cfg.length_eq, cfg.dim_hidden, out=self.repair_tagger_pos_embedding.weight)
            tagger_layer = nn.TransformerDecoderLayer(
                d_model=cfg.dim_hidden,
                nhead=cfg.num_heads,
                dim_feedforward=self.repair_dec_pf_dim,
                dropout=self.repair_dropout_p,
            )
            self.repair_tagger_decoder = nn.TransformerDecoder(tagger_layer, num_layers=self.repair_layers)
            self.repair_tagger_dropout = nn.Dropout(self.repair_dropout_p)
            self.tagger_action_head = nn.Linear(cfg.dim_hidden, NUM_TAGGER_ACTIONS)

            # Editor: separate conditional Transformer for local token synthesis.
            self.repair_generator_tok_embedding = nn.Embedding(cfg.output_dim, cfg.dim_hidden)
            self.repair_generator_pos_embedding = nn.Embedding(cfg.length_eq, cfg.dim_hidden)
            if self.repair_use_sinusoidal_embeddings:
                self.create_sinusoidal_embeddings(cfg.length_eq, cfg.dim_hidden, out=self.repair_generator_pos_embedding.weight)
            generator_layer = nn.TransformerDecoderLayer(
                d_model=cfg.dim_hidden,
                nhead=cfg.num_heads,
                dim_feedforward=self.repair_dec_pf_dim,
                dropout=self.repair_dropout_p,
            )
            self.repair_generator_decoder = nn.TransformerDecoder(generator_layer, num_layers=self.repair_layers)
            self.repair_generator_dropout = nn.Dropout(self.repair_dropout_p)
            # Shared editor trunk with action-specific heads.
            self.repair_editor_fuse = nn.Sequential(
                nn.Linear(cfg.dim_hidden * 2, cfg.dim_hidden),
                nn.GELU(),
                nn.Dropout(self.repair_dropout_p),
            )
            self.generator_fc_out_by_action = nn.ModuleDict({
                str(int(ACT_INSERT)): nn.Linear(cfg.dim_hidden, cfg.output_dim),
                str(int(ACT_REWRITE_SUBTREE)): nn.Linear(cfg.dim_hidden, cfg.output_dim),
            })
            self.repair_replace_head = nn.Linear(cfg.dim_hidden, cfg.output_dim)
            self.repair_delete_head = nn.Linear(cfg.dim_hidden, cfg.output_dim)

            # Editor-only conditioning
            self.repair_action_embedding = nn.Embedding(NUM_REPAIR_ACTIONS, cfg.dim_hidden)
            self.repair_mask_embedding = nn.Parameter(torch.zeros(cfg.dim_hidden))
            nn.init.normal_(self.repair_mask_embedding, mean=0.0, std=0.02)

            self.repair_fixed_hole_slots: int = int(max(2, int(self.repair_direct_rewrite_max_nodes)))

    def create_sinusoidal_embeddings(self, n_pos, dim, out):
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        sin_part = torch.tensor(np.sin(position_enc[:, 0::2]), dtype=out.dtype, device=out.device)
        cos_part = torch.tensor(np.cos(position_enc[:, 1::2]), dtype=out.dtype, device=out.device)
        with torch.no_grad():
            out[:, 0::2].copy_(sin_part)
            out[:, 1::2].copy_(cos_part)
            out.requires_grad_(False)

    def _get_cached_causal_mask(self, length: int, device: torch.device):
        key = (str(device), int(length))
        cached = self._causal_mask_cache.get(key, None)
        if cached is None or cached.device != device:
            future = torch.triu(torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1)
            cached = torch.zeros((length, length), device=device, dtype=torch.float32)
            cached = cached.masked_fill(future, float("-inf"))
            self._causal_mask_cache[key] = cached
        return cached

    def make_trg_mask(self, trg):
        """Create masks for nn.TransformerDecoder.

        Returns:
            trg_key_padding_mask: [B, L] bool, True for PAD positions (ignored in attention)
            trg_causal_mask:      [L, L] float, 0 for allowed, -inf for disallowed (future) positions
        """
        # trg: [B, L] token ids
        _, L = trg.shape
        device = trg.device

        # Key padding mask: True for PAD tokens
        trg_key_padding_mask = (trg == self.trg_pad_idx)

        # Causal mask: disallow attending to future positions (j > i)
        trg_causal_mask = self._get_cached_causal_mask(L, device)

        return trg_key_padding_mask, trg_causal_mask

    def decoder_output(self, trg_, encoder_output, trg_key_padding_mask, trg_causal_mask):
        # encoder_output is [B, S, D]; permute to [S, B, D] for nn.TransformerDecoder
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            encoder_output.permute(1, 0, 2),
            trg_causal_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
        )
        return output

    def encode_only(self, batch):
        """Encode the numeric point-feature sequence (same condition as AR), without decoding."""
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, :(size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)
        encoder_input = torch.cat((src_x, src_y), dim=-1)
        enc_out = self.MultiModalEncoder(encoder_input)
        return enc_out

    def forward_with_enc(self, batch, num_epochs=None, current_epoch=None, batch_idx=0):
        """Same as forward(), but also returns encoder output for optional repair head."""
        # Prepare Inputs
        b = batch[0].permute(0, 2, 1)
        size = b.shape[-1]
        src_x = b[:, :, :(size - 1)]
        src_y = b[:, :, -1].unsqueeze(2)

        # Prepare Target & Masks
        trg = batch[1].long()
        pos_ids = self._pos_idx_eq[: batch[1].shape[1] - 1].unsqueeze(0).expand(batch[1].shape[0], -1)
        pos = self.pos_embedding(pos_ids)
        te = self.tok_embedding(trg[:, :-1])
        trg_ = self.dropout(te + pos)
        trg_mask1, trg_mask2 = self.make_trg_mask(trg[:, :-1])

        # Prepare Dataset Features
        encoder_input = torch.cat((src_x, src_y), dim=-1)

        # --- Encoder ---
        enc_out = self.MultiModalEncoder(encoder_input)

        # --- Symbolic Decoding (AR) ---
        output_logits = self.fc_out(self.decoder_output(trg_, enc_out, trg_mask1, trg_mask2))
        # Hard-mask forbidden special tokens so AR can never emit them.
        output_logits = self._mask_forbidden_output_logits(output_logits)
        return output_logits, trg, enc_out

    def forward(self, batch, num_epochs=None, current_epoch=None, batch_idx=0):
        output_logits, trg, _ = self.forward_with_enc(
            batch,
            num_epochs=num_epochs,
            current_epoch=current_epoch,
            batch_idx=batch_idx,
        )
        return output_logits, trg

    def compute_loss(self, output_logits, trg):
        """Cross-entropy loss for token generation (symbolic regression)."""
        trg_flat = trg[:, 1:].contiguous().view(-1)
        output_flat = output_logits.permute(1, 0, 2).contiguous().view(-1, output_logits.shape[-1])
        loss = self.CrossEntropy_Loss(output_flat, trg_flat)
        return loss

    def _ar_greedy_free_run_init(self,
                                 enc_out: torch.Tensor,
                                 seq_len: int,
                                 allowed_leaf_ids_batch: Optional[List[List[int]]] = None,
                                 return_step_logits: bool = False):
        """AR constrained greedy free-running decode.

        Args:
            enc_out: [B,N,D]
            seq_len: packed sequence length including S/F/PAD.
            allowed_leaf_ids_batch: optional per-sample leaf whitelist.
            return_step_logits: if True, also return per-step next-token logits and an active mask
                for the actually decoded body positions. This lets us attach an auxiliary AR loss under
                the model's own greedy prefixes while still reusing the same decode pass.

        Returns:
            If return_step_logits is False:
                packed tokens [B,L] as [S] body [F] PAD...
            Else:
                (packed_tokens [B,L], step_logits [B,max_body_len,V], step_active_mask [B,max_body_len])
        """
        self._maybe_init_vocab_helper()
        if self._repair_helper is None:
            B = enc_out.shape[0]
            zeros = torch.zeros((B, seq_len), dtype=torch.long, device=enc_out.device)
            if return_step_logits:
                max_body_len = max(0, seq_len - 2)
                step_logits = torch.zeros((B, max_body_len, self.cfg.output_dim), dtype=enc_out.dtype, device=enc_out.device)
                step_mask = torch.zeros((B, max_body_len), dtype=torch.bool, device=enc_out.device)
                return zeros, step_logits, step_mask
            return zeros

        helper = self._repair_helper
        B = enc_out.shape[0]
        max_body_len = max(0, seq_len - 2)

        # tokens for decoder input (prefix): start + body tokens
        tokens = torch.full((B, 1), int(helper.start_id), dtype=torch.long, device=enc_out.device)
        need = torch.ones((B,), dtype=torch.long, device=enc_out.device)
        finished = torch.zeros((B,), dtype=torch.bool, device=enc_out.device)

        bodies: List[List[int]] = [[] for _ in range(B)]
        step_logits_list: List[torch.Tensor] = []
        step_active_list: List[torch.Tensor] = []

        for i in range(max_body_len):
            if finished.all():
                break
            cur_len = tokens.shape[1]
            pos_ids = self._pos_idx_eq[:cur_len].to(enc_out.device).unsqueeze(0).expand(B, -1)
            pos = self.pos_embedding(pos_ids)
            te = self.tok_embedding(tokens)
            trg_ = self.dropout(te + pos)
            trg_mask1 = (tokens == self.trg_pad_idx)
            trg_mask2 = self._get_cached_causal_mask(cur_len, enc_out.device)
            out = self.fc_out(self.decoder_output(trg_, enc_out, trg_mask1, trg_mask2))  # [cur_len, B, V]
            next_logits = out[cur_len - 1]  # [B, V]
            next_logits = self._mask_forbidden_output_logits(next_logits)

            active_now = (~finished).clone()
            if return_step_logits:
                step_logits_list.append(next_logits)
                step_active_list.append(active_now)

            rem = max_body_len - i - 1

            for b in range(B):
                if finished[b]:
                    continue

                allowed_leaf = allowed_leaf_ids_batch[
                    b] if allowed_leaf_ids_batch is not None else helper.default_leaf_ids
                # allowed token ids based on remaining length
                allowed: List[int] = []
                n = int(need[b].item())

                # leaf: new_need = n-1
                if (n - 1) <= rem:
                    allowed.extend(list(allowed_leaf) if allowed_leaf else helper.default_leaf_ids)

                # unary: new_need = n
                if helper.unary_ids and (n <= rem):
                    allowed.extend(helper.unary_ids)

                # binary: new_need = n+1
                if helper.binary_ids and ((n + 1) <= rem):
                    allowed.extend(helper.binary_ids)

                # strip specials
                specials = {helper.pad_id, helper.start_id, helper.finish_id}
                allowed = [int(t) for t in allowed if int(t) not in specials and int(helper.arity(int(t))) >= 0]
                if not allowed:
                    tok = helper._random_const_leaf()
                else:
                    # argmax among allowed
                    logits_b = next_logits[b]
                    mask = torch.full_like(logits_b, float('-inf'))
                    mask[allowed] = 0.0
                    tok = int((logits_b + mask).argmax(dim=-1).item())

                # Safety: never accept invalid/sentinel tokens inside decoded bodies.
                a_tok = int(helper.arity(tok))
                if a_tok < 0:
                    tok = int(helper._random_const_leaf())
                    a_tok = int(helper.arity(tok))

                bodies[b].append(tok)
                # update need
                need[b] = need[b] - 1 + a_tok
                if int(need[b].item()) == 0:
                    finished[b] = True

            # append chosen tokens to tokens (for unfinished samples; finished append PAD placeholder)
            next_tok_vec = torch.full((B, 1), int(helper.pad_id), dtype=torch.long, device=enc_out.device)
            for b in range(B):
                if active_now[b]:
                    next_tok_vec[b, 0] = int(bodies[b][-1])
            tokens = torch.cat([tokens, next_tok_vec], dim=1)

        packed_rows = [helper.pack(body, max_len=seq_len) for body in bodies]
        packed = torch.tensor(packed_rows, dtype=torch.long, device=enc_out.device)
        if not return_step_logits:
            return packed

        if step_logits_list:
            step_logits = torch.stack(step_logits_list, dim=1)
            step_mask = torch.stack(step_active_list, dim=1)
        else:
            step_logits = torch.zeros((B, 0, self.cfg.output_dim), dtype=enc_out.dtype, device=enc_out.device)
            step_mask = torch.zeros((B, 0), dtype=torch.bool, device=enc_out.device)
        steps_done = int(step_logits.shape[1])
        if steps_done < max_body_len:
            pad_steps = max_body_len - steps_done
            step_logits = torch.cat(
                [step_logits, torch.zeros((B, pad_steps, self.cfg.output_dim), dtype=step_logits.dtype, device=step_logits.device)],
                dim=1,
            )
            step_mask = torch.cat(
                [step_mask, torch.zeros((B, pad_steps), dtype=torch.bool, device=step_mask.device)],
                dim=1,
            )
        return packed, step_logits, step_mask

    def _ar_beam_free_run_init(self,
                               enc_out: torch.Tensor,
                               seq_len: int,
                               beam_size: int,
                               allowed_leaf_ids_batch: Optional[List[List[int]]] = None) -> torch.Tensor:
        """Syntax-constrained AR beam search helper. Training free-run starts now use greedy decode; this is kept for inference-side utilities / compatibility.

        The highest-scoring finished beam (or, if none finishes, the best partial beam) is packed into
        [S] body [F] PAD... and used as the corrupted starting point for oracle-chain construction.
        """
        self._maybe_init_vocab_helper()
        if self._repair_helper is None:
            B = enc_out.shape[0]
            return torch.zeros((B, seq_len), dtype=torch.long, device=enc_out.device)

        helper = self._repair_helper
        B = int(enc_out.shape[0])
        max_body_len = max(0, int(seq_len) - 2)
        beam_size = int(max(1, beam_size))
        out_seq = torch.full((B, seq_len), int(helper.pad_id), dtype=torch.long, device=enc_out.device)

        for b in range(B):
            allowed_leaf = allowed_leaf_ids_batch[b] if allowed_leaf_ids_batch is not None else helper.default_leaf_ids
            if not allowed_leaf:
                allowed_leaf = helper.default_leaf_ids

            beams: List[Dict[str, Any]] = [dict(body=[], need=1, logp=0.0, rank_score=0.0, finished=False)]
            best_finished: Optional[Dict[str, Any]] = None

            for i in range(max_body_len):
                active = [st for st in beams if not bool(st['finished'])]
                if not active:
                    break

                cur_len = 1 + i
                toks = torch.full((len(active), cur_len), int(helper.pad_id), dtype=torch.long, device=enc_out.device)
                toks[:, 0] = int(helper.start_id)
                for row, st in enumerate(active):
                    body = st['body']
                    if body:
                        toks[row, 1:1 + len(body)] = torch.tensor(body, dtype=torch.long, device=enc_out.device)

                enc_rep = enc_out[b:b + 1].expand(len(active), -1, -1).contiguous()
                pos_ids = self._pos_idx_eq[:cur_len].to(enc_out.device).unsqueeze(0).expand(len(active), -1)
                pos = self.pos_embedding(pos_ids)
                te = self.tok_embedding(toks)
                trg_ = self.dropout(te + pos)
                trg_mask1, trg_mask2 = self.make_trg_mask(toks)
                out = self.fc_out(self.decoder_output(trg_, enc_rep, trg_mask1, trg_mask2))
                next_logits = self._mask_forbidden_output_logits(out[cur_len - 1])

                rem = max_body_len - i - 1
                cand_pool: List[Dict[str, Any]] = [
                    dict(
                        body=list(st['body']),
                        need=int(st['need']),
                        logp=float(st['logp']),
                        rank_score=float(st.get('rank_score', float(st['logp']) / max(1, len(st['body'])))),
                        finished=True,
                    )
                    for st in beams if bool(st['finished'])
                ]

                for row, st in enumerate(active):
                    n = int(st['need'])
                    allowed: List[int] = []
                    if (n - 1) <= rem:
                        allowed.extend(list(allowed_leaf))
                    if helper.unary_ids and (n <= rem):
                        allowed.extend(helper.unary_ids)
                    if helper.binary_ids and ((n + 1) <= rem):
                        allowed.extend(helper.binary_ids)
                    specials = {helper.pad_id, helper.start_id, helper.finish_id}
                    allowed = sorted({int(t) for t in allowed if int(t) not in specials and int(helper.arity(int(t))) >= 0})
                    if not allowed:
                        allowed = [int(helper._random_const_leaf())]

                    logits_b = next_logits[row]
                    allow_idx = torch.as_tensor(allowed, device=logits_b.device, dtype=torch.long)
                    allow_logits = logits_b.index_select(0, allow_idx)
                    topk = int(min(beam_size, int(allow_idx.numel())))
                    vals, ids = torch.topk(allow_logits, k=topk, dim=0, largest=True, sorted=True)

                    for v, j in zip(vals.detach().cpu().tolist(), ids.detach().cpu().tolist()):
                        tok = int(allow_idx[int(j)].item())
                        a_tok = int(helper.arity(tok))
                        if a_tok < 0:
                            continue
                        new_need = int(n - 1 + a_tok)
                        if new_need < 0:
                            continue
                        new_body = list(st['body']) + [tok]
                        finished = bool(new_need == 0)
                        new_logp = float(st['logp']) + float(v)
                        new_state = dict(
                            body=new_body,
                            need=new_need,
                            logp=new_logp,
                            rank_score=float(new_logp) / float(max(1, len(new_body))),
                            finished=finished,
                        )
                        cand_pool.append(new_state)
                        if finished and helper.validate_body(new_body):
                            if best_finished is None:
                                best_finished = dict(new_state)
                            else:
                                better = False
                                if float(new_state['rank_score']) > float(best_finished.get('rank_score', -1e18)) + 1e-12:
                                    better = True
                                elif abs(float(new_state['rank_score']) - float(best_finished.get('rank_score', -1e18))) <= 1e-12:
                                    if float(new_state['logp']) > float(best_finished.get('logp', -1e18)) + 1e-12:
                                        better = True
                                if better:
                                    best_finished = dict(new_state)

                if not cand_pool:
                    break

                cand_pool.sort(key=lambda st: (-float(st.get('rank_score', float(st['logp']) / max(1, len(st['body'])))), -float(st['logp']), len(st['body'])))
                beams = cand_pool[:beam_size]

            if best_finished is not None:
                best_body = list(best_finished['body'])
            else:
                best_state = None
                for st in beams:
                    if best_state is None:
                        best_state = st
                        continue
                    cur_rank = float(st.get('rank_score', float(st['logp']) / max(1, len(st['body']))))
                    best_rank = float(best_state.get('rank_score', float(best_state['logp']) / max(1, len(best_state['body']))))
                    if cur_rank > best_rank + 1e-12:
                        best_state = st
                    elif abs(cur_rank - best_rank) <= 1e-12 and float(st['logp']) > float(best_state['logp']) + 1e-12:
                        best_state = st
                best_body = list(best_state['body']) if best_state is not None else []

            if not helper.validate_body(best_body):
                best_body = [int(helper._random_const_leaf())]
            packed = helper.pack(best_body, max_len=seq_len)
            out_seq[b] = torch.tensor(packed, dtype=torch.long, device=enc_out.device)

        return out_seq

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_dataset_size = len(self.trainer.datamodule.train_dataloader().dataset)
            batch_size = self.cfg.batch_size
            self.steps_per_epoch = math.ceil(train_dataset_size / batch_size)
            self.total_steps = self.steps_per_epoch * self.cfg.epochs

    def _log_repair_metrics(self, prefix: str, out: Optional[Dict[str, Any]], *, on_step: bool, on_epoch: bool, prog_bar: bool = False) -> None:
        if out is None:
            return
        key_specs = [
            ('repair_loss', f'{prefix}_repair_loss'),
            ('tagger_loss', f'{prefix}_repair_tagger_loss'),
            ('generator_loss', f'{prefix}_repair_editor_loss'),
            ('repair_states_total', f'{prefix}_repair_states_total'),
            ('repair_ar_state_count', f'{prefix}_repair_ar_state_count'),
            ('repair_synth_state_count', f'{prefix}_repair_synth_state_count'),
            ('repair_rollout_state_count', f'{prefix}_repair_rollout_state_count'),
            ('repair_keep_self_state_count', f'{prefix}_repair_keep_self_state_count'),
            ('repair_tagger_overall_acc', f'{prefix}_repair_tagger_acc'),
            ('repair_action_prop_keep', f'{prefix}_repair_action_prop_keep'),
            ('repair_action_prop_replace', f'{prefix}_repair_action_prop_replace'),
            ('repair_action_prop_delete', f'{prefix}_repair_action_prop_delete'),
            ('repair_action_prop_rewrite', f'{prefix}_repair_action_prop_rewrite'),
            ('repair_action_prop_insert', f'{prefix}_repair_action_prop_insert'),
            ('repair_action_acc_keep', f'{prefix}_repair_action_acc_keep'),
            ('repair_action_acc_replace', f'{prefix}_repair_action_acc_replace'),
            ('repair_action_acc_delete', f'{prefix}_repair_action_acc_delete'),
            ('repair_action_acc_rewrite', f'{prefix}_repair_action_acc_rewrite'),
            ('repair_action_acc_insert', f'{prefix}_repair_action_acc_insert'),
            ('repair_editor_loss_keep', f'{prefix}_repair_editor_loss_keep'),
            ('repair_editor_loss_replace', f'{prefix}_repair_editor_loss_replace'),
            ('repair_editor_loss_delete', f'{prefix}_repair_editor_loss_delete'),
            ('repair_editor_loss_rewrite', f'{prefix}_repair_editor_loss_rewrite'),
            ('repair_editor_loss_insert', f'{prefix}_repair_editor_loss_insert'),
            ('repair_editor_acc_keep', f'{prefix}_repair_editor_acc_keep'),
            ('repair_editor_acc_replace', f'{prefix}_repair_editor_acc_replace'),
            ('repair_editor_acc_delete', f'{prefix}_repair_editor_acc_delete'),
            ('repair_editor_acc_rewrite', f'{prefix}_repair_editor_acc_rewrite'),
            ('repair_editor_acc_insert', f'{prefix}_repair_editor_acc_insert'),
            ('repair_editor_token_count_keep', f'{prefix}_repair_editor_token_count_keep'),
            ('repair_editor_token_count_replace', f'{prefix}_repair_editor_token_count_replace'),
            ('repair_editor_token_count_delete', f'{prefix}_repair_editor_token_count_delete'),
            ('repair_editor_token_count_rewrite', f'{prefix}_repair_editor_token_count_rewrite'),
            ('repair_editor_token_count_insert', f'{prefix}_repair_editor_token_count_insert'),
            ('repair_editor_exact_overall', f'{prefix}_repair_editor_exact_overall'),
            ('repair_editor_exact_replace', f'{prefix}_repair_editor_exact_replace'),
            ('repair_editor_exact_delete', f'{prefix}_repair_editor_exact_delete'),
            ('repair_editor_exact_rewrite', f'{prefix}_repair_editor_exact_rewrite'),
            ('repair_editor_exact_insert', f'{prefix}_repair_editor_exact_insert'),
            ('repair_overfit_train_cache_size', f'{prefix}_repair_overfit_train_cache_size'),
            ('repair_overfit_val_cache_size', f'{prefix}_repair_overfit_val_cache_size'),
            ('repair_pairs_total', f'{prefix}_repair_pairs_total'),
            ('free_run_chain_success', f'{prefix}_repair_free_run_chain_success'),
            ('synth_chain_success', f'{prefix}_repair_synth_chain_success'),
            ('oracle_empty_chain_ratio', f'{prefix}_repair_oracle_empty_chain_ratio'),
        ]
        for src_key, log_key in key_specs:
            if src_key in out:
                self.log(log_key, out[src_key], on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar if src_key in {'repair_loss', 'tagger_loss', 'generator_loss'} else False)

    def _repair_valid_action_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None
        tokens = self._truncate_after_F(tokens)
        B, L = tokens.shape
        mask = torch.zeros((B, L, NUM_TAGGER_ACTIONS), dtype=torch.bool, device=tokens.device)
        mask[:, :, int(ACT_KEEP)] = True
        body_rows = [helper.extract_body(row) for row in tokens.detach().cpu().tolist()]
        for b, body in enumerate(body_rows):
            for i, tok in enumerate(body):
                pos = 1 + int(i)
                if pos >= L:
                    break
                ar = int(helper.arity(int(tok)))
                if ar == 0:
                    legal = (ACT_KEEP, ACT_REPLACE, ACT_INSERT)
                elif ar > 0:
                    legal = (ACT_KEEP, ACT_REPLACE, ACT_DELETE_SUBTREE, ACT_REWRITE_SUBTREE)
                else:
                    legal = (ACT_KEEP,)
                for act in legal:
                    mask[b, pos, int(act)] = True
        return mask

    def _repair_overfit_clone_batch(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu().clone()
        if isinstance(obj, dict):
            return {k: self._repair_overfit_clone_batch(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._repair_overfit_clone_batch(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._repair_overfit_clone_batch(v) for v in obj)
        return obj

    def _repair_overfit_to_device(self, obj, device: torch.device):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._repair_overfit_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._repair_overfit_to_device(v, device) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._repair_overfit_to_device(v, device) for v in obj)
        return obj

    def _repair_overfit_select_batch(self, batch, batch_idx: int, split: str):
        if not self.repair_overfit_debug_enable:
            return batch
        device = next(self.parameters()).device
        if split == 'train':
            cache = self._repair_overfit_train_cache
            target_n = self.repair_overfit_debug_train_batches
        else:
            if self.repair_overfit_debug_use_train_cache_for_val and len(self._repair_overfit_train_cache) > 0:
                source = self._repair_overfit_train_cache[int(batch_idx) % len(self._repair_overfit_train_cache)]
                return self._repair_overfit_to_device(source, device)
            cache = self._repair_overfit_val_cache
            target_n = self.repair_overfit_debug_val_batches
        if len(cache) < target_n:
            cache.append(self._repair_overfit_clone_batch(batch))
            return batch
        source = cache[int(batch_idx) % len(cache)]
        return self._repair_overfit_to_device(source, device)

    def training_step(self, batch, batch_idx):
        batch = self._repair_overfit_select_batch(batch, batch_idx=batch_idx, split='train')
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad(set_to_none=True)

        if self.repair_only and self.repair_enable and self.repair_train:
            # Repair-only mode should not backprop through the shared condition encoder,
            # but the resulting encoder states still need to participate as ordinary
            # tensors in downstream autograd for the repair heads. Using `torch.no_grad()`
            # keeps enc_out detached while preserving ordinary tensor semantics for downstream
            # Transformer backward passes.
            with torch.no_grad():
                enc_out = self.encode_only(batch)
            enc_out = enc_out
            out = self.compute_repair_loss_from_batch(batch, enc_out=enc_out, output_logits=None, batch_idx=batch_idx)
            if out is None:
                return
            repair_loss = out["repair_loss"]
            loss = repair_loss

            self._log_repair_metrics("train", out, on_step=True, on_epoch=True)
            self.log("train_loss", loss, on_step=True, on_epoch=True)

            if not bool(loss.requires_grad):
                # This can happen when no valid repair supervision pair is built for the batch
                # (e.g. oracle chain returns empty and auxiliary losses are disabled).
                self.log("train_repair_skipped_no_grad", torch.tensor(1.0, device=loss.device), on_step=True, on_epoch=True)
                return

            self.manual_backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()
            return

        # Default: AR training (unchanged), plus optional repair loss
        output_logits, trg, enc_out = self.forward_with_enc(
            batch,
            num_epochs=self.num_epochs,
            current_epoch=self.current_epoch,
            batch_idx=batch_idx,
        )

        ce_loss = self.compute_loss(output_logits, trg)
        loss = ce_loss

        repair_loss = None
        repair_out = None
        if self.repair_enable and self.repair_train:
            repair_out = self.compute_repair_loss_from_batch(batch, enc_out=enc_out, output_logits=output_logits, batch_idx=batch_idx if self.training else None)
            if repair_out is not None:
                repair_loss = repair_out.get("repair_loss", None)
                if repair_loss is not None:
                    loss = loss + repair_loss

        self.log("train_ce_loss", ce_loss, on_step=True, on_epoch=True)
        if repair_out is not None:
            self._log_repair_metrics("train", repair_out, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        if not bool(loss.requires_grad):
            self.log("train_skipped_no_grad", torch.tensor(1.0, device=loss.device), on_step=True, on_epoch=True)
            return

        self.manual_backward(loss)
        optimizer.step()
        if scheduler:
            scheduler.step()

    def validation_step(self, batch, batch_idx):
        batch = self._repair_overfit_select_batch(batch, batch_idx=batch_idx, split='val')
        output_logits, trg, enc_out = self.forward_with_enc(
            batch,
            num_epochs=self.num_epochs,
            current_epoch=self.current_epoch,
        )

        ce_loss = self.compute_loss(output_logits, trg)
        loss = ce_loss

        repair_loss = None
        repair_out = None
        if self.repair_enable and self.repair_train:
            repair_out = self.compute_repair_loss_from_batch(batch, enc_out=enc_out, output_logits=output_logits, batch_idx=None)
            if repair_out is not None:
                repair_loss = repair_out.get("repair_loss", None)
                if repair_loss is not None:
                    loss = loss + repair_loss

        self.log("val_ce_loss", ce_loss, on_epoch=True, prog_bar=True)
        if repair_out is not None:
            self._log_repair_metrics("val", repair_out, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def get_lr_lambda(self, current_step):
        total_steps = getattr(self, "total_steps", 1)
        progress = float(current_step) / float(max(1, total_steps))
        lr_mult = 1.0 - 0.9 * (1.0 - math.cos(math.pi * 0.5 * progress))
        return lr_mult

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": 1e-3},
                {"params": no_decay_params, "weight_decay": 0.0}
            ],
            lr=self.cfg.lr
        )
        scheduler = {
            'scheduler': LambdaLR(optimizer, self.get_lr_lambda),
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [scheduler]

    # -------------------------------------------------------------------------
    # Vocab caching (from data.py EditSRDataset.word2id / id2word)
    # -------------------------------------------------------------------------
    def _maybe_init_vocab_helper(self):
        if self._vocab_ready:
            return
        if not (isinstance(self.word2id, dict) and isinstance(self.id2word, dict) and len(self.word2id) > 0):
            return

        # --- Extend vocab with explicit repair tag tokens (global-skeleton) ---
        # These tokens are ONLY used inside the repair generator input; they are excluded from grammar leaf pools.
        next_id = int(max(self.word2id.values())) + 1 if len(self.word2id) > 0 else 0
        for tok in [self.repair_tag_repl_str, self.repair_tag_insert_str]:
            if tok not in self.word2id:
                self.word2id[tok] = int(next_id)
                self.id2word[int(next_id)] = str(tok)
                next_id += 1

        # Sanity: embedding/output head must be large enough for the extended vocab.
        # cfg.output_dim is fixed at model init; if it is too small, training/inference will crash later.
        max_id = int(max(self.word2id.values())) if len(self.word2id) > 0 else -1
        if int(getattr(self.cfg, "output_dim", max_id + 1)) <= max_id:
            raise ValueError(
                f"cfg.output_dim={int(getattr(self.cfg, 'output_dim'))} is smaller than vocab max_id+1={max_id + 1}. "
                f"Please increase architecture.output_dim in config.yaml to >= {max_id + 1} to use repair tag tokens."
            )

        self._repair_helper = PrefixRepairHelper(self.word2id, self.id2word)
        self._repair_helper.corruption_action_weights = {
            'leaf_replace': float(self.repair_corruption_weight_replace),
            'op_replace': float(self.repair_corruption_weight_replace),
            'subtree_delete': float(self.repair_corruption_weight_insert),
            'leaf_insert': float(self.repair_corruption_weight_delete),
            'subtree_replace': float(self.repair_corruption_weight_rewrite),
        }
        # cache ids
        self.repair_tag_repl_id = int(self.word2id.get(self.repair_tag_repl_str))
        self.repair_tag_insert_id = int(self.word2id.get(self.repair_tag_insert_str))

        # Cache forbidden output ids (e.g. <repl:0>/<insert:0>) so logits can be masked everywhere.
        # This is a hard safety net: even if a candidate pool forgets to filter, generation cannot emit them.
        self._forbidden_output_ids = []
        for _tok, _tid in self.word2id.items():
            if isinstance(_tok, str) and _tok.startswith("<") and _tok.endswith(">"):
                self._forbidden_output_ids.append(int(_tid))
        self._forbidden_output_ids = sorted(set(self._forbidden_output_ids))
        self._vocab_ready = True

    def _mask_forbidden_output_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Hard-mask logits for forbidden special tokens (e.g. explicit repair tags like <repl:0>/<insert:0>).

        Applied in BOTH training and inference, for both AR and repair generator heads, to guarantee these tokens
        can never be emitted even if a candidate pool / constrained decode forgets to filter them.
        """
        self._maybe_init_vocab_helper()
        ids = getattr(self, "_forbidden_output_ids", None)
        if not ids:
            return logits
        if logits is None or logits.numel() == 0:
            return logits
        V = int(logits.shape[-1])
        valid_ids = [int(i) for i in ids if 0 <= int(i) < V]
        if not valid_ids:
            return logits

        # Use a large negative finite value (fp16-safe) instead of -inf.
        mask_val = float(self.forbidden_logit_value)

        out = logits.clone()
        idx = torch.as_tensor(valid_ids, device=out.device, dtype=torch.long)
        out.index_fill_(-1, idx, mask_val)
        return out

    def _try_cache_vocab_from_trainer(self):
        """Try to pull vocab from datamodule's dataset (train/val/test)."""
        if self._vocab_ready:
            return
        if not hasattr(self, "trainer") or self.trainer is None:
            return

        # Try train loader first
        candidates = []
        try:
            candidates.append(self.trainer.datamodule.train_dataloader().dataset)
        except Exception:
            pass
        try:
            candidates.append(self.trainer.datamodule.val_dataloader().dataset)
        except Exception:
            pass
        try:
            candidates.append(self.trainer.datamodule.test_dataloader().dataset)
        except Exception:
            pass

        for ds in candidates:
            # unwrap ConcatDataset-like
            if ds is None:
                continue
            if hasattr(ds, "datasets") and len(getattr(ds, "datasets")) > 0:
                ds0 = ds.datasets[0]
            else:
                ds0 = ds
            if hasattr(ds0, "word2id") and hasattr(ds0, "id2word"):
                self.word2id = dict(ds0.word2id)
                self.id2word = dict(ds0.id2word)
                self._maybe_init_vocab_helper()
                return

    def _beam_length_penalty(self, length: int, alpha: float) -> float:
        """GNMT-style length penalty used for beam ranking."""
        length = int(max(1, length))
        alpha = float(max(0.0, alpha))
        if alpha <= 0.0:
            return 1.0
        return float(((5.0 + float(length)) ** alpha) / ((5.0 + 1.0) ** alpha))

    def _length_normalized_score(self, raw_logp: float, length: int, alpha: float) -> float:
        return float(raw_logp) / float(self._beam_length_penalty(length=length, alpha=alpha))

    def on_fit_start(self):
        self._try_cache_vocab_from_trainer()
        self._repair_trace_reset_file()

    def on_test_start(self):
        self._try_cache_vocab_from_trainer()

    def on_predict_start(self):
        self._try_cache_vocab_from_trainer()

    def _repair_action_name(self, action: Optional[int]) -> str:
        mapping = {
            int(ACT_KEEP): 'KEEP',
            int(ACT_REPLACE): 'REPLACE',
            int(ACT_DELETE_SUBTREE): 'DELETE_SUBTREE',
            int(ACT_REWRITE_SUBTREE): 'REWRITE_SUBTREE',
            int(ACT_INSERT): 'INSERT',
        }
        if action is None:
            return ''
        return str(mapping.get(int(action), str(action)))

    def _repair_token_str(self, tok: Optional[int]) -> str:
        if tok is None:
            return ''
        self._maybe_init_vocab_helper()
        if self.id2word is None:
            return str(int(tok))
        return str(self.id2word.get(int(tok), str(int(tok))))

    def _repair_body_token_str(self, body: Optional[List[int]]) -> str:
        return ' '.join(self._repair_token_str(int(t)) for t in (body or []))

    def _repair_trace_is_enabled(self) -> bool:
        if not bool(self.repair_trace_enable):
            return False
        if bool(self.repair_trace_train_only) and not bool(self.training):
            return False
        if int(getattr(self, 'global_rank', 0)) != 0:
            return False
        if self.repair_trace_cases_per_batch <= 0 or self.repair_trace_max_cases <= 0:
            return False
        return True

    def _repair_trace_get_csv_path(self) -> str:
        if self._repair_trace_csv_path is not None:
            return str(self._repair_trace_csv_path)
        out = str(self.repair_trace_output_csv)
        if os.path.isabs(out):
            path = out
        else:
            root = None
            try:
                if getattr(self, 'trainer', None) is not None:
                    root = getattr(self.trainer, 'default_root_dir', None) or getattr(self.trainer, 'log_dir', None)
            except Exception:
                root = None
            if not root:
                from src.EditSR.project_paths import project_path
                root = str(project_path())
            path = os.path.join(str(root), out)
        self._repair_trace_csv_path = path
        return path

    def _repair_trace_reset_file(self) -> None:
        self._repair_trace_cases_written = 0
        self._repair_trace_rows_written = 0
        self._repair_trace_header_written = False
        self._repair_trace_csv_path = None
        if not bool(self.repair_trace_enable):
            return
        if int(getattr(self, 'global_rank', 0)) != 0:
            return
        path = self._repair_trace_get_csv_path()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._repair_trace_fieldnames)
            writer.writeheader()
        self._repair_trace_header_written = True

    def _repair_trace_write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows or not self._repair_trace_is_enabled():
            return
        path = self._repair_trace_get_csv_path()
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        need_header = (not self._repair_trace_header_written) or (not os.path.exists(path))
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self._repair_trace_fieldnames)
            if need_header:
                writer.writeheader()
                self._repair_trace_header_written = True
            for row in rows:
                clean = {k: '' for k in self._repair_trace_fieldnames}
                for k, v in row.items():
                    if k not in clean:
                        continue
                    if isinstance(v, (dict, list, tuple)):
                        clean[k] = json.dumps(v, ensure_ascii=False)
                    elif v is None:
                        clean[k] = ''
                    else:
                        clean[k] = v
                writer.writerow(clean)
        self._repair_trace_rows_written += int(len(rows))

    def _repair_trace_frontier_labels_json(self, frontier: List[Dict[str, Any]]) -> str:
        serial = []
        for e in frontier:
            serial.append({
                'root_idx': int(e.get('root_idx', -1)),
                'action': self._repair_action_name(e.get('action', None)),
                'span_start': int(e.get('span_start', -1)),
                'span_end': int(e.get('span_end', -1)),
                'prev_root_token': int(e['prev_root_token']) if e.get('prev_root_token', None) is not None else None,
                'prev_root_token_str': self._repair_token_str(e.get('prev_root_token', None)),
                'target_subtree': [int(x) for x in (e.get('target_subtree', []) or [])],
                'target_subtree_tokens': self._repair_body_token_str(e.get('target_subtree', []) or []),
            })
        return json.dumps(serial, ensure_ascii=False)

    def _repair_trace_rows_from_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        gt_body = [int(x) for x in payload.get('gt_body', [])]
        source_body = [int(x) for x in payload.get('source_body', [])]
        sampled_cur_body = [int(x) for x in payload.get('sampled_cur_body', [])]
        frontier = list(payload.get('frontier', []))
        meta = dict(payload.get('source_meta', {}) or {})
        base = {
            'trace_id': str(payload.get('trace_id', '')),
            'split': str(payload.get('split', 'train')),
            'epoch': int(payload.get('epoch', -1)),
            'global_step': int(payload.get('global_step', -1)),
            'batch_idx': int(payload.get('batch_idx', -1)),
            'source_batch_idx': int(payload.get('src_b', -1)),
            'source_type': str(payload.get('source_type', '')),
            'gt_body_ids': gt_body,
            'gt_body_tokens': self._repair_body_token_str(gt_body),
            'source_body_ids': source_body,
            'source_body_tokens': self._repair_body_token_str(source_body),
            'state_body_ids': sampled_cur_body,
            'state_body_tokens': self._repair_body_token_str(sampled_cur_body),
            'chain_length': int(payload.get('chain_length', 0)),
            'frontier_count': int(len(frontier)),
            'frontier_labels_json': self._repair_trace_frontier_labels_json(frontier),
        }
        rows: List[Dict[str, Any]] = [{
            **base,
            'row_type': 'summary',
            'row_index': 0,
            'distribution_name': str(payload.get('chain_sampling_distribution', '')),
            'sampled_value': payload.get('chain_sampled_index', ''),
            'extra_json': {
                'desired_depth_distribution': meta.get('desired_depth_distribution', ''),
                'desired_depth_sample': meta.get('desired_depth_sample', None),
                'realized_corruption_depth': meta.get('realized_corruption_depth', None),
                'resample_attempts_max': meta.get('resample_attempts_max', None),
                'resample_attempts_used': meta.get('resample_attempts_used', meta.get('resample_attempts', None)),
                'attempt_chain_lengths': meta.get('attempt_chain_lengths', None),
                'rollout_steps': meta.get('rollout_steps', None),
                'supervision_mode': meta.get('supervision_mode', None),
                'training_states_added': meta.get('training_states_added', None),
            },
        }]
        for i, rec in enumerate(meta.get('corruption_chain', []) or []):
            prev_body = [int(x) for x in rec.get('prev_body', [])]
            cur_body = [int(x) for x in rec.get('cur_body', [])]
            prev_subtree = [int(x) for x in rec.get('prev_subtree', [])]
            rows.append({
                **base,
                'row_type': 'corruption_step',
                'row_index': int(i),
                'distribution_name': str(meta.get('desired_depth_distribution', '')),
                'sampled_value': int(meta.get('desired_depth_sample', 0) or 0),
                'before_body_ids': prev_body,
                'before_body_tokens': self._repair_body_token_str(prev_body),
                'after_body_ids': cur_body,
                'after_body_tokens': self._repair_body_token_str(cur_body),
                'action_name': str(rec.get('forward_op', '')),
                'root_idx': int(rec.get('root_idx', -1)),
                'prev_span': list(rec.get('prev_span', [])),
                'cur_span': list(rec.get('cur_span', [])),
                'edited_subtree_ids': prev_subtree,
                'edited_subtree_tokens': self._repair_body_token_str(prev_subtree),
                'target_token_id': int(rec['prev_root_token']) if rec.get('prev_root_token', None) is not None else '',
                'target_token_str': self._repair_token_str(rec.get('prev_root_token', None)),
            })
        for i, rec in enumerate(payload.get('oracle_chain', []) or []):
            cur_body = [int(x) for x in rec.get('cur_body', [])]
            prev_body = [int(x) for x in rec.get('prev_body', [])]
            target_subtree = [int(x) for x in (rec.get('target_subtree', []) or [])]
            edit_content = [int(x) for x in (rec.get('edit_content', []) or [])]
            rows.append({
                **base,
                'row_type': 'oracle_chain_step',
                'row_index': int(i),
                'before_body_ids': cur_body,
                'before_body_tokens': self._repair_body_token_str(cur_body),
                'after_body_ids': prev_body,
                'after_body_tokens': self._repair_body_token_str(prev_body),
                'action_name': self._repair_action_name(rec.get('rev_action', None)),
                'action_id': int(rec.get('rev_action', -1)),
                'root_idx': int(rec.get('root_idx', -1)),
                'span_start': int(rec.get('span_start', -1)),
                'span_end': int(rec.get('span_end', -1)),
                'target_token_id': int(rec['prev_root_token']) if rec.get('prev_root_token', None) is not None else '',
                'target_token_str': self._repair_token_str(rec.get('prev_root_token', None)),
                'target_subtree_ids': target_subtree,
                'target_subtree_tokens': self._repair_body_token_str(target_subtree),
                'edited_subtree_ids': edit_content,
                'edited_subtree_tokens': self._repair_body_token_str(edit_content),
            })
        for i, e in enumerate(frontier):
            tgt = [int(x) for x in (e.get('target_subtree', []) or [])]
            rows.append({
                **base,
                'row_type': 'frontier_edit',
                'row_index': int(i),
                'action_name': self._repair_action_name(e.get('action', None)),
                'action_id': int(e.get('action', -1)),
                'root_idx': int(e.get('root_idx', -1)),
                'span_start': int(e.get('span_start', -1)),
                'span_end': int(e.get('span_end', -1)),
                'target_token_id': int(e['prev_root_token']) if e.get('prev_root_token', None) is not None else '',
                'target_token_str': self._repair_token_str(e.get('prev_root_token', None)),
                'target_subtree_ids': tgt,
                'target_subtree_tokens': self._repair_body_token_str(tgt),
            })
        return rows

    # -------------------------------------------------------------------------
    # Repair head: masking up to F (ignore PAD and stop at F)
    # -------------------------------------------------------------------------
    def _truncate_after_F(self, tokens: torch.Tensor) -> torch.Tensor:
        """把每行第一个结束符(F)之后的 token 全部置为 PAD（保持 shape 不变）。"""
        self._maybe_init_vocab_helper()
        if self._repair_helper is None:
            pad_id, finish_id = 0, 2
        else:
            pad_id, finish_id = int(self._repair_helper.pad_id), int(self._repair_helper.finish_id)

        if tokens.dim() != 2:
            raise ValueError("tokens must be [B, L]")

        out = tokens.clone()
        B, L = out.shape
        # 找到每行第一个 F 的位置；如果没有 F，则 idx = L-1（等价于不截断）
        is_f = (out == finish_id)
        has_f = is_f.any(dim=1)
        first_f = torch.argmax(is_f.long(), dim=1)  # 若无 F -> 0，需要下面用 has_f 修正
        first_f = torch.where(has_f, first_f, torch.full_like(first_f, L - 1))

        ar = torch.arange(L, device=out.device).unsqueeze(0)  # [1,L]
        after = ar > first_f.unsqueeze(1)  # [B,L] True 表示 F 后面
        out[after] = pad_id
        return out

    def _repair_token_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B,L] 布尔 mask：哪些位置参与 repair loss（从 S 开始到第一个 F 为止，忽略 PAD）。"""
        self._maybe_init_vocab_helper()
        if self._repair_helper is None:
            pad_id, finish_id = 0, 2
        else:
            pad_id, finish_id = int(self._repair_helper.pad_id), int(self._repair_helper.finish_id)

        if tokens.dim() != 2:
            raise ValueError("tokens must be [B, L]")

        B, L = tokens.shape
        ar = torch.arange(L, device=tokens.device).unsqueeze(0)  # [1,L]

        is_pad = (tokens == pad_id)
        is_f = (tokens == finish_id)
        has_f = is_f.any(dim=1)
        first_f = torch.argmax(is_f.long(), dim=1)
        first_f = torch.where(has_f, first_f, torch.full_like(first_f, L - 1))

        # <= first_f 的位置都算在内；最后再把 PAD 排掉
        mask = ar <= first_f.unsqueeze(1)
        mask = mask & (~is_pad)
        return mask

    def _ar_logits_to_pred_tokens(self, output_logits: torch.Tensor, trg_tokens: torch.Tensor) -> torch.Tensor:
        """Cheap init from AR teacher-forced logits, but decoded with prefix-grammar constraints."""
        self._maybe_init_vocab_helper()
        if self._repair_helper is None:
            return self._truncate_after_F(trg_tokens.clone())

        helper = self._repair_helper
        B, L = trg_tokens.shape

        # allowed leaves per sample: constants + variables present in GT body
        allowed_leaf_ids_batch = []
        trg_trunc = self._truncate_after_F(trg_tokens)
        for b in range(B):
            body = helper.extract_body(trg_trunc[b].detach().cpu().tolist())
            allowed_leaf_ids_batch.append(helper.allowed_leaf_ids_from_body(body))

        pred_tokens = constrained_decode_batch_from_position_logits(
            output_logits=output_logits,
            helper=helper,
            seq_len=L,
            allowed_leaf_ids_batch=allowed_leaf_ids_batch,
        )
        return self._truncate_after_F(pred_tokens)

    def _repair_tagger_hidden(self, tokens: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """Tagger hidden states from a dedicated conditional Transformer."""
        tokens = self._truncate_after_F(tokens)
        _, L = tokens.shape
        device = tokens.device

        pad_id = int(self._repair_helper.pad_id) if self._repair_helper is not None else int(self.trg_pad_idx)

        pos_ids = self._pos_idx_eq[:L].to(device).unsqueeze(0).expand(tokens.shape[0], -1)
        pos_emb = self.repair_tagger_pos_embedding(pos_ids)
        tok_emb = self.repair_tagger_tok_embedding(tokens)
        emb = self.repair_tagger_dropout(tok_emb + pos_emb)
        tgt_key_padding_mask = (tokens == pad_id)

        hidden = self.repair_tagger_decoder(
            emb.permute(1, 0, 2),
            enc_out.permute(1, 0, 2),
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return hidden

    def repair_tagger_logits(self, tokens: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """Predict dense 5-way rooted edit logits for each body position.

        The Tagger is implemented as a conditional TransformerDecoder without a causal mask, so
        self-attention is bidirectional over the current expression. A linear head then scores
        {KEEP, REPLACE, DELETE_SUBTREE, REWRITE_SUBTREE, INSERT} at every position.
        """
        hidden = self._repair_tagger_hidden(tokens=tokens, enc_out=enc_out)
        action_logits = self.tagger_action_head(hidden).permute(1, 0, 2).contiguous()
        if bool(self.repair_tagger_use_action_mask):
            valid_mask = self._repair_valid_action_mask(tokens)
            action_logits = action_logits.masked_fill(~valid_mask, float(self.forbidden_logit_value))
        return action_logits

    def _build_repair_infill_attn_mask(self, tokens: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        """Build global-skeleton infill/prefix-LM self-attention mask for the repair generator.

        Goal: let span tokens (holes) see BOTH left + right known context, while keeping AR within the span.

        Mask semantics (self-attn):
          - known query positions attend only to known keys (bidirectional)
          - hole query positions attend to:
              * all known keys (both sides)
              * hole keys at positions <= query (AR within holes)
          - PAD keys are always masked

        Returns:
            attn_mask: float tensor of shape [B * H, L, L] filled with 0 (allowed) or -inf (blocked)
        """
        if tokens.dim() != 2 or hole_mask.dim() != 2:
            raise ValueError("tokens and hole_mask must be [B,L]")
        if tokens.shape != hole_mask.shape:
            raise ValueError("tokens and hole_mask must have the same shape")

        self._maybe_init_vocab_helper()
        pad_id = int(self._repair_helper.pad_id) if self._repair_helper is not None else int(self.trg_pad_idx)

        B, L = tokens.shape
        device = tokens.device

        is_pad = (tokens == pad_id)  # [B,L]
        hole = hole_mask.bool()
        known = ~hole

        # Base: always mask PAD keys
        disallow = is_pad.unsqueeze(1).expand(B, L, L).clone()  # [B,L,L]

        # Known queries: cannot attend to any hole keys
        disallow |= (known.unsqueeze(2) & hole.unsqueeze(1))

        # Hole queries: cannot attend to FUTURE hole keys (keep AR inside holes)
        future = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)  # [L,L]
        disallow |= (hole.unsqueeze(2) & hole.unsqueeze(1) & future.unsqueeze(0))

        attn = torch.zeros((B, L, L), device=device, dtype=torch.float32)
        attn = attn.masked_fill(disallow, float("-inf"))

        # Expand to [B*H, L, L] for per-sample masking in MultiheadAttention
        try:
            H = int(self.repair_generator_decoder.layers[0].self_attn.num_heads)
        except Exception:
            H = int(getattr(self.cfg, "num_heads", 1))
        attn = attn.repeat_interleave(max(1, H), dim=0)
        return attn

    def _repair_generator_hidden(
            self,
            tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: Optional[torch.Tensor] = None,
            hole_mask: Optional[torch.Tensor] = None,
            tgt_mask: Optional[torch.Tensor] = None,
            replace_holes_with_mask_embedding: bool = True,
    ) -> torch.Tensor:
        """Single-stream editor hidden states from the shared editor trunk."""
        tokens = self._truncate_after_F(tokens)
        _, L = tokens.shape
        device = tokens.device

        pad_id = int(self._repair_helper.pad_id) if self._repair_helper is not None else int(self.trg_pad_idx)

        pos_ids = self._pos_idx_eq[:L].to(device).unsqueeze(0).expand(tokens.shape[0], -1)
        pos_emb = self.repair_generator_pos_embedding(pos_ids)

        tok_emb = self.repair_generator_tok_embedding(tokens)
        if hole_mask is not None and bool(replace_holes_with_mask_embedding):
            tok_emb = tok_emb.clone()
            tok_emb[hole_mask] = self.repair_mask_embedding

        emb = tok_emb + pos_emb
        if action_ids is not None:
            emb = emb + self.repair_action_embedding(action_ids.clamp(0, NUM_REPAIR_ACTIONS - 1))
        emb = self.repair_generator_dropout(emb)

        tgt_key_padding_mask = (tokens == pad_id)

        hidden = self.repair_generator_decoder(
            emb.permute(1, 0, 2),
            enc_out.permute(1, 0, 2),
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return hidden.permute(1, 0, 2).contiguous()

    def _repair_editor_fused_hidden(
            self,
            *,
            orig_tokens: torch.Tensor,
            edit_tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: Optional[torch.Tensor] = None,
            hole_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dual-stream editor encoding.

        The editor sees both the original packed sequence and the edited skeleton sequence.
        Both streams use the same Transformer trunk and are fused position-wise.
        """
        if hole_mask is None:
            hole_mask = torch.zeros_like(edit_tokens, dtype=torch.bool)

        edit_tgt_mask = None
        if bool(self.repair_gen_use_infill_mask):
            edit_tgt_mask = self._build_repair_infill_attn_mask(tokens=edit_tokens, hole_mask=hole_mask)

        replace_holes = not bool(self.repair_gen_use_tag_tokens)
        embed_action_ids = action_ids if bool(self.repair_gen_use_action_embedding) else None

        orig_hidden = self._repair_generator_hidden(
            tokens=orig_tokens,
            enc_out=enc_out,
            action_ids=None,
            hole_mask=torch.zeros_like(orig_tokens, dtype=torch.bool),
            tgt_mask=None,
            replace_holes_with_mask_embedding=False,
        )
        edit_hidden = self._repair_generator_hidden(
            tokens=edit_tokens,
            enc_out=enc_out,
            action_ids=embed_action_ids,
            hole_mask=hole_mask,
            tgt_mask=edit_tgt_mask,
            replace_holes_with_mask_embedding=replace_holes,
        )
        fused = self.repair_editor_fuse(torch.cat([orig_hidden, edit_hidden], dim=-1))
        return fused

    def repair_generator_logits(
            self,
            *,
            orig_tokens: torch.Tensor,
            edit_tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: Optional[torch.Tensor] = None,
            hole_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Token logits for INSERT / REWRITE generation heads under dual-stream conditioning."""
        fused = self._repair_editor_fused_hidden(
            orig_tokens=orig_tokens,
            edit_tokens=edit_tokens,
            enc_out=enc_out,
            action_ids=action_ids,
            hole_mask=hole_mask,
        )
        if action_ids is None:
            action_ids = torch.zeros_like(edit_tokens, dtype=torch.long)
        head_action_ids = action_ids.clamp(0, NUM_REPAIR_ACTIONS - 1)
        logits_insert = self.generator_fc_out_by_action[str(int(ACT_INSERT))](fused)
        logits_rewrite = self.generator_fc_out_by_action[str(int(ACT_REWRITE_SUBTREE))](fused)
        is_insert = (head_action_ids == int(ACT_INSERT)).unsqueeze(-1)
        token_logits = torch.where(is_insert, logits_insert, logits_rewrite)
        token_logits = self._mask_forbidden_output_logits(token_logits)
        return token_logits

    def repair_replace_delete_logits(
            self,
            *,
            orig_tokens: torch.Tensor,
            edit_tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: torch.Tensor,
            hole_mask: torch.Tensor,
            root_positions: torch.Tensor,
            action_label: int,
    ) -> torch.Tensor:
        """Single-token classification logits for REPLACE / DELETE actions."""
        fused = self._repair_editor_fused_hidden(
            orig_tokens=orig_tokens,
            edit_tokens=edit_tokens,
            enc_out=enc_out,
            action_ids=action_ids,
            hole_mask=hole_mask,
        )
        root_positions = root_positions.clamp(min=0, max=fused.shape[1] - 1)
        idx = root_positions.view(-1, 1, 1).expand(-1, 1, fused.shape[-1])
        root_hidden = fused.gather(1, idx).squeeze(1)
        if int(action_label) == int(ACT_REPLACE):
            logits = self.repair_replace_head(root_hidden)
            logits = self._mask_forbidden_output_logits(logits)
            return logits
        if int(action_label) == int(ACT_DELETE_SUBTREE):
            logits = self.repair_delete_head(root_hidden)
            logits = self._mask_forbidden_output_logits(logits)
            self._maybe_init_vocab_helper()
            helper = self._repair_helper
            if helper is not None:
                leaf_mask = torch.zeros((logits.shape[-1],), dtype=torch.bool, device=logits.device)
                for v in helper.global_leaf_ids():
                    if 0 <= int(v) < logits.shape[-1]:
                        leaf_mask[int(v)] = True
                neg_inf = torch.finfo(logits.dtype).min
                logits = logits.masked_fill(~leaf_mask.unsqueeze(0), neg_inf)
            return logits
        raise ValueError(f'Unsupported single-token editor action: {action_label}')

    def compute_repair_loss(self, repair_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Token-level CE over positions up to F (inclusive), ignoring PAD.

        Does NOT project-to-valid. Assumes target is already valid.
        """
        self._maybe_init_vocab_helper()
        pad_id = int(self._repair_helper.pad_id) if self._repair_helper is not None else 0

        target_tokens = self._truncate_after_F(target_tokens)
        mask = self._repair_token_mask(target_tokens)
        mask[:, 0] = False  # don't train on S

        B, L, V = repair_logits.shape
        loss_flat = F.cross_entropy(
            repair_logits.reshape(B * L, V),
            target_tokens.reshape(B * L),
            reduction="none",
            ignore_index=int(pad_id),
        ).view(B, L)

        denom = mask.float().sum().clamp_min(1.0)
        return (loss_flat * mask.float()).sum() / denom

        # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Deterministic action-space synthetic chain construction (GT -> imperfect)
    # -------------------------------------------------------------------------
    @staticmethod
    def _oracle_split_children(body: Tuple[int, ...], helper: PrefixRepairHelper) -> List[Tuple[int, ...]]:
        """Split a valid prefix subtree into its ordered child subtrees."""
        if not body:
            return []
        a = int(helper.arity(int(body[0])))
        if a <= 0:
            return []
        out: List[Tuple[int, ...]] = []
        j = 1
        body_list = list(body)
        for _ in range(a):
            e = int(helper.subtree_end(body_list, j))
            out.append(tuple(int(x) for x in body_list[j:e]))
            j = e
        return out

    @staticmethod
    def _oracle_first_leaf(body: Tuple[int, ...], helper: PrefixRepairHelper) -> int:
        """Return the first leaf token in preorder (representative leaf)."""
        for tok in body:
            if int(helper.arity(int(tok))) == 0:
                return int(tok)
        return int(body[-1]) if body else int(helper._random_const_leaf())

    @staticmethod
    def _oracle_min_scaffold_size(body: Tuple[int, ...], helper: PrefixRepairHelper) -> int:
        """Minimum valid size of a scaffold rooted at body[0]."""
        if not body:
            return 1
        a = int(helper.arity(int(body[0])))
        if a <= 0:
            return 1
        return int(1 + a)

    def _oracle_canonical_scaffold(self, body: Tuple[int, ...], helper: PrefixRepairHelper, budget: int) -> Tuple[int, ...]:
        """Unique maximal preorder-prefix-closed scaffold within a size budget.

        Omitted descendants are replaced by representative leaves so the result stays prefix-valid.
        """
        budget = int(max(1, budget))
        body = tuple(int(x) for x in body)
        if (not body) or int(helper.arity(int(body[0]))) <= 0 or len(body) <= 1:
            return body

        min_needed = self._oracle_min_scaffold_size(body, helper)
        if budget < min_needed:
            # Fallback: use the smallest valid rooted scaffold.
            root = int(body[0])
            out = [root]
            for ch in self._oracle_split_children(body, helper):
                out.append(self._oracle_first_leaf(ch, helper))
            return tuple(out)

        root = int(body[0])
        children = self._oracle_split_children(body, helper)
        out: List[int] = [root]
        rem = int(budget - 1)
        for i, ch in enumerate(children):
            later = len(children) - i - 1
            min_later = later  # one representative leaf per remaining child slot
            avail = int(rem - min_later)
            if int(helper.arity(int(ch[0]))) == 0:
                out.append(int(ch[0]))
                rem -= 1
                continue
            child_min = self._oracle_min_scaffold_size(ch, helper)
            if avail >= child_min:
                scf = self._oracle_canonical_scaffold(ch, helper, avail)
                out.extend(list(scf))
                rem -= len(scf)
            else:
                out.append(self._oracle_first_leaf(ch, helper))
                rem -= 1
        return tuple(out)

    def _parse_prefix_addr_map(self, body: List[int], helper: PrefixRepairHelper) -> Dict[Tuple[int, ...], Dict[str, int]]:
        """Map rooted child-index paths to subtree spans in a prefix-valid body.

        Each entry stores the start/end offsets of the subtree currently addressed by that rooted
        path. The root subtree is addressed by ().
        """
        body_list = [int(x) for x in body]
        amap: Dict[Tuple[int, ...], Dict[str, int]] = {}
        if not body_list:
            return amap

        def _dfs(start: int, path: Tuple[int, ...]) -> int:
            end = int(helper.subtree_end(body_list, start))
            tok = int(body_list[start])
            amap[tuple(path)] = {'start': int(start), 'end': int(end), 'tok': int(tok)}
            ar = int(helper.arity(tok))
            if ar > 0:
                j = int(start + 1)
                for k in range(ar):
                    j = int(_dfs(j, tuple(path) + (int(k),)))
            return int(end)

        _dfs(0, ())
        return amap

    def _same_root_children_structure(
            self,
            lhs: List[int],
            rhs: List[int],
            helper: PrefixRepairHelper,
    ) -> bool:
        lhs = [int(t) for t in lhs]
        rhs = [int(t) for t in rhs]
        if (not lhs) or (not rhs):
            return False
        a_l = int(helper.arity(int(lhs[0])))
        a_r = int(helper.arity(int(rhs[0])))
        if a_l <= 0 or a_l != a_r:
            return False
        lhs_ch = self._oracle_split_children(tuple(lhs), helper)
        rhs_ch = self._oracle_split_children(tuple(rhs), helper)
        if len(lhs_ch) != len(rhs_ch):
            return False
        return all(list(cx) == list(cy) for cx, cy in zip(lhs_ch, rhs_ch))

    def _classify_local_repair_action(
            self,
            *,
            cur_sub: List[int],
            prev_sub: List[int],
            helper: PrefixRepairHelper,
    ) -> Tuple[int, Optional[List[int]], Optional[int]]:
        """Classify a local cur->prev rooted transition under the 5-action repair space.

        Returns:
            (action_id, target_subtree, prev_root_token)

        The classification is budget-aware: a generic REWRITE_SUBTREE is only emitted
        when the local subtree pair fits inside the rewrite budget. Larger structural
        changes must be decomposed upstream instead of being collapsed into a single
        oversized rewrite supervision label.
        """
        cur_sub = [int(t) for t in cur_sub]
        prev_sub = [int(t) for t in prev_sub]
        if (not cur_sub) or (not prev_sub) or cur_sub == prev_sub:
            return ACT_KEEP, None, None

        a_cur = int(helper.arity(int(cur_sub[0])))
        a_prev = int(helper.arity(int(prev_sub[0])))
        rewrite_budget = int(max(2, int(self.repair_direct_rewrite_max_nodes)))

        # leaf -> leaf   OR   non-leaf with identical children and same-arity root relabel
        if a_cur == 0 and a_prev == 0:
            return ACT_REPLACE, None, int(prev_sub[0])
        if a_cur > 0 and a_prev > 0 and a_cur == a_prev:
            if self._same_root_children_structure(cur_sub, prev_sub, helper):
                return ACT_REPLACE, None, int(prev_sub[0])

        # subtree -> leaf collapse
        if a_cur > 0 and a_prev == 0:
            return ACT_DELETE_SUBTREE, None, int(prev_sub[0])

        # leaf -> closed subtree growth
        if a_cur == 0 and a_prev > 0:
            return ACT_INSERT, list(prev_sub), None

        # generic structural rewrite between non-leaf subtrees, but only if the
        # local pair fits in the configured rewrite budget.
        if a_cur > 0 and a_prev > 0 and max(len(cur_sub), len(prev_sub)) <= rewrite_budget:
            return ACT_REWRITE_SUBTREE, list(prev_sub), None

        return ACT_KEEP, None, None

    def _oracle_apply_action_at_path(
            self,
            body: List[int],
            *,
            path: Tuple[int, ...],
            action: int,
            helper: PrefixRepairHelper,
            target_token: Optional[int] = None,
            target_subtree: Optional[List[int]] = None,
            max_body_len: int,
    ) -> Optional[Tuple[List[int], int, int]]:
        """Apply one forward action at a rooted path.

        Returns (new_body, root_idx, end_idx) or None if the path/action is invalid.
        """
        amap = self._parse_prefix_addr_map(body, helper)
        node = amap.get(tuple(path), None)
        if node is None:
            return None
        s, e = int(node['start']), int(node['end'])
        cur_sub = [int(x) for x in body[s:e]]
        if not cur_sub:
            return None
        cur_root = int(cur_sub[0])
        cur_arity = int(helper.arity(cur_root))

        new_body: List[int]
        if int(action) == ACT_REPLACE:
            if target_token is None:
                return None
            tgt_tok = int(target_token)
            if int(helper.arity(tgt_tok)) != int(cur_arity):
                return None
            new_body = list(body)
            new_body[s] = tgt_tok

        elif int(action) == ACT_DELETE_SUBTREE:
            if target_token is None:
                return None
            tgt_tok = int(target_token)
            if cur_arity <= 0 or int(helper.arity(tgt_tok)) != 0:
                return None
            new_body = list(body[:s]) + [tgt_tok] + list(body[e:])

        elif int(action) == ACT_REWRITE_SUBTREE:
            tgt = [int(x) for x in (target_subtree or [])]
            if (not tgt) or cur_arity <= 0 or int(helper.arity(int(tgt[0]))) <= 0:
                return None
            new_body = list(body[:s]) + tgt + list(body[e:])

        elif int(action) == ACT_INSERT:
            tgt = [int(x) for x in (target_subtree or [])]
            if (not tgt) or cur_arity != 0 or int(helper.arity(int(tgt[0]))) <= 0:
                return None
            new_body = list(body[:s]) + tgt + list(body[e:])

        else:
            return None

        if (not new_body) or len(new_body) > int(max_body_len) or (not helper.validate_body(new_body)):
            return None
        return list(new_body), int(s), int(e)

    def _oracle_build_chain(
            self,
            *,
            cur_body: List[int],
            gt_body: List[int],
            helper: PrefixRepairHelper,
            max_body_len: int,
    ) -> List[Dict[str, Any]]:
        """Construct an exact current->GT single-edit supervision chain.

        The chain is built *directly* in the repair direction (current -> GT), so each
        returned step already lives in the training action space and respects the local
        rewrite budget enforced by the oracle solver.
        """
        src0 = [int(x) for x in cur_body[:max_body_len]]
        tgt0 = [int(x) for x in gt_body[:max_body_len]]
        if (not src0) or (not tgt0) or (not helper.validate_body(src0)) or (not helper.validate_body(tgt0)):
            return []
        if src0 == tgt0:
            return []

        rewrite_budget = int(max(2, int(self.repair_direct_rewrite_max_nodes)))
        lam = float(self.repair_frontier_lambda)

        memo: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[float, List[Dict[str, Any]]]] = {}
        visiting: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()
        INF = float('inf')

        solve_stats: Dict[str, int] = {
            'calls': 0,
            'memo_hits': 0,
            'cycle_hits': 0,
            'max_depth': 0,
            'scf_self_skip': 0,
            'scf_target_skip': 0,
            'scf_cycle_skip': 0,
            'rho_self_skip': 0,
            'rho_target_skip': 0,
            'rho_cycle_skip': 0,
            'debug_prints': 0,
        }
        debug_print_limit = 0

        def _oracle_dbg(msg: str) -> None:
            if solve_stats['debug_prints'] >= debug_print_limit:
                return
            if int(getattr(self, 'global_rank', 0)) == 0:
                print(msg)
            solve_stats['debug_prints'] += 1

        def _prefixed(plan: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for st in plan:
                out.append({**st, 'path': (int(k),) + tuple(st['path'])})
            return out

        def _cost_rep(x: Tuple[int, ...], y: Tuple[int, ...]) -> float:
            return 0.0 if int(x[0]) == int(y[0]) else 1.0

        def _cost_del(x: Tuple[int, ...], y_leaf: Tuple[int, ...]) -> float:
            return float(1.0 + lam * max(0, len(x) - 1))

        def _cost_ins(x_leaf: Tuple[int, ...], y: Tuple[int, ...]) -> float:
            return float(1.0 + lam * max(0, len(y) - 1))

        def _cost_rw(x: Tuple[int, ...], y: Tuple[int, ...]) -> float:
            return float(1.0 + lam * len(y))

        def solve(x: Tuple[int, ...], y: Tuple[int, ...], depth: int = 0) -> Tuple[float, List[Dict[str, Any]]]:
            x = tuple(int(t) for t in x)
            y = tuple(int(t) for t in y)
            key = (x, y)

            solve_stats['calls'] += 1
            solve_stats['max_depth'] = max(int(solve_stats['max_depth']), int(depth))

            if key in memo:
                solve_stats['memo_hits'] += 1
                return memo[key]
            if key in visiting:
                solve_stats['cycle_hits'] += 1
                _oracle_dbg(
                    f"[oracle.solve/cycle] depth={depth} len(x)={len(x)} len(y)={len(y)} "
                    f"x_head={x[:12]} y_head={y[:12]}"
                )
                return INF, []
            if x == y:
                memo[key] = (0.0, [])
                return memo[key]
            if (not x) or (not y):
                memo[key] = (INF, [])
                return memo[key]

            visiting.add(key)
            try:
                ax = int(helper.arity(int(x[0])))
                ay = int(helper.arity(int(y[0])))
                candidates: List[Tuple[float, List[Dict[str, Any]]]] = []

                if ax == 0 and ay == 0:
                    rep_cost = _cost_rep(x, y)
                    rep_plan = []
                    if int(x[0]) != int(y[0]):
                        rep_plan.append(dict(path=(), action=ACT_REPLACE, target_token=int(y[0]), target_subtree=None))
                    candidates.append((float(rep_cost), rep_plan))

                if ax > 0 and ay > 0 and ax == ay:
                    xch = self._oracle_split_children(x, helper)
                    ych = self._oracle_split_children(y, helper)
                    rep_cost = _cost_rep(x, y)
                    rep_plan: List[Dict[str, Any]] = []
                    if int(x[0]) != int(y[0]):
                        rep_plan.append(dict(path=(), action=ACT_REPLACE, target_token=int(y[0]), target_subtree=None))
                    ok = True
                    for k, (cx, cy) in enumerate(zip(xch, ych)):
                        cc, cp = solve(cx, cy, depth + 1)
                        if math.isinf(cc):
                            ok = False
                            break
                        rep_cost += float(cc)
                        rep_plan.extend(_prefixed(cp, k))
                    if ok:
                        candidates.append((float(rep_cost), rep_plan))

                if ax > 0 and ay == 0:
                    leaf_tok = int(y[0])
                    del_cost = _cost_del(x, y)
                    del_plan = [dict(path=(), action=ACT_DELETE_SUBTREE, target_token=leaf_tok, target_subtree=None)]
                    candidates.append((float(del_cost), del_plan))

                if ax == 0 and ay > 0:
                    if len(y) <= rewrite_budget:
                        ins_cost = _cost_ins(x, y)
                        ins_plan = [dict(path=(), action=ACT_INSERT, target_token=None,
                                         target_subtree=[int(t) for t in y])]
                        candidates.append((float(ins_cost), ins_plan))
                    else:
                        scf = tuple(int(t) for t in self._oracle_canonical_scaffold(y, helper, rewrite_budget))
                        if scf and helper.validate_body(list(scf)) and int(helper.arity(int(scf[0]))) > 0 and scf != y:
                            c2, p2 = solve(scf, y, depth + 1)
                            if not math.isinf(c2):
                                bridge_cost = _cost_ins(x, scf)
                                bridge_plan = [dict(path=(), action=ACT_INSERT, target_token=None,
                                                    target_subtree=[int(t) for t in scf])] + p2
                                candidates.append((float(bridge_cost + c2), bridge_plan))

                if ax > 0 and ay > 0 and max(len(x), len(y)) <= rewrite_budget:
                    rw_cost = _cost_rw(x, y)
                    rw_plan = [dict(path=(), action=ACT_REWRITE_SUBTREE, target_token=None,
                                    target_subtree=[int(t) for t in y])]
                    candidates.append((float(rw_cost), rw_plan))

                if ax > 0 and ay > 0 and max(len(x), len(y)) > rewrite_budget:
                    scf = tuple(int(t) for t in self._oracle_canonical_scaffold(y, helper, rewrite_budget))
                    if scf and helper.validate_body(list(scf)):
                        if scf == x:
                            solve_stats['scf_self_skip'] += 1
                            _oracle_dbg(
                                f"[oracle.solve/scf_self_skip] depth={depth} len(x)={len(x)} len(y)={len(y)} "
                                f"x_head={x[:12]} y_head={y[:12]}"
                            )
                        elif scf == y:
                            solve_stats['scf_target_skip'] += 1
                        elif (scf, y) in visiting:
                            solve_stats['scf_cycle_skip'] += 1
                            _oracle_dbg(
                                f"[oracle.solve/scf_cycle_skip] depth={depth} len(scf)={len(scf)} len(y)={len(y)} "
                                f"scf_head={scf[:12]} y_head={y[:12]}"
                            )
                        else:
                            c2, p2 = solve(scf, y, depth + 1)
                            if not math.isinf(c2):
                                if int(helper.arity(int(scf[0]))) > 0:
                                    bridge_cost = _cost_rw(x, scf)
                                    bridge_plan = [dict(path=(), action=ACT_REWRITE_SUBTREE, target_token=None,
                                                        target_subtree=[int(t) for t in scf])] + p2
                                else:
                                    bridge_cost = _cost_del(x, scf)
                                    bridge_plan = [dict(path=(), action=ACT_DELETE_SUBTREE, target_token=int(scf[0]),
                                                        target_subtree=None)] + p2
                                candidates.append((float(bridge_cost + c2), bridge_plan))

                if not candidates:
                    memo[key] = (INF, [])
                    return memo[key]
                best = min(candidates, key=lambda z: (float(z[0]), len(z[1])))
                memo[key] = (float(best[0]), list(best[1]))
                return memo[key]
            finally:
                visiting.discard(key)

        total_cost, rel_plan = solve(tuple(src0), tuple(tgt0), depth=0)
        solve_bad = (
            int(solve_stats['cycle_hits'])
            + int(solve_stats['scf_self_skip'])
            + int(solve_stats['scf_cycle_skip'])
            + int(solve_stats['rho_self_skip'])
            + int(solve_stats['rho_cycle_skip'])
        )
        if solve_bad > 0:
            _oracle_dbg(
                "[oracle.solve/summary] "
                f"calls={solve_stats['calls']} memo_hits={solve_stats['memo_hits']} cycle_hits={solve_stats['cycle_hits']} "
                f"max_depth={solve_stats['max_depth']} scf_self_skip={solve_stats['scf_self_skip']} "
                f"scf_target_skip={solve_stats['scf_target_skip']} scf_cycle_skip={solve_stats['scf_cycle_skip']} "
                f"rho_self_skip={solve_stats['rho_self_skip']} rho_target_skip={solve_stats['rho_target_skip']} "
                f"rho_cycle_skip={solve_stats['rho_cycle_skip']}"
            )
            try:
                stage = 'train' if bool(self.training) else 'val'
                self.log_dict(
                    {
                        f'{stage}/oracle_solve_calls': torch.tensor(float(solve_stats['calls']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_memo_hits': torch.tensor(float(solve_stats['memo_hits']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_cycle_hits': torch.tensor(float(solve_stats['cycle_hits']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_max_depth': torch.tensor(float(solve_stats['max_depth']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_scf_self_skip': torch.tensor(float(solve_stats['scf_self_skip']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_scf_cycle_skip': torch.tensor(float(solve_stats['scf_cycle_skip']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_rho_self_skip': torch.tensor(float(solve_stats['rho_self_skip']), dtype=torch.float32, device=self.device),
                        f'{stage}/oracle_solve_rho_cycle_skip': torch.tensor(float(solve_stats['rho_cycle_skip']), dtype=torch.float32, device=self.device),
                    },
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=False,
                )
            except Exception:
                pass
        if math.isinf(total_cost):
            return []

        forward_steps: List[Dict[str, Any]] = []
        body = list(src0)
        for st in rel_plan:
            res = self._oracle_apply_action_at_path(
                body=body,
                path=tuple(st['path']),
                action=int(st['action']),
                helper=helper,
                target_token=st.get('target_token', None),
                target_subtree=st.get('target_subtree', None),
                max_body_len=max_body_len,
            )
            if res is None:
                return []
            new_body, root_idx, end_idx = res
            forward_steps.append(dict(
                before_body=list(body),
                after_body=list(new_body),
                action=int(st['action']),
                path=tuple(st['path']),
                root_idx_before=int(root_idx),
                span_start=int(root_idx),
                span_end=int(end_idx),
                edited_before=list(body[root_idx:end_idx]),
                edited_after=list(new_body[root_idx: helper.subtree_end(new_body, root_idx)]),
                target_token=st.get('target_token', None),
                target_subtree=list(st['target_subtree']) if st.get('target_subtree', None) is not None else None,
            ))
            body = list(new_body)

        if body != tgt0:
            return []

        chain: List[Dict[str, Any]] = []
        for rec in forward_steps:
            target_subtree = list(rec['target_subtree']) if rec.get('target_subtree', None) is not None else None
            target_token = int(rec['target_token']) if rec.get('target_token', None) is not None else None
            edit_content: List[int] = []
            if target_subtree is not None:
                edit_content = list(target_subtree)
            elif target_token is not None:
                edit_content = [int(target_token)]
            chain.append(dict(
                cur_body=list(rec['before_body']),
                prev_body=list(rec['after_body']),
                rev_action=int(rec['action']),
                root_idx=int(rec['root_idx_before']),
                span_start=int(rec['span_start']),
                span_end=int(rec['span_end']),
                path=tuple(rec['path']),
                target_subtree=target_subtree,
                prev_root_token=target_token,
                edit_content=edit_content,
            ))
        return chain


    def _removed_compute_repair_loss_from_batch(
            self,
            batch,
            enc_out: torch.Tensor,
            output_logits: Optional[torch.Tensor] = None,
    ):
        raise RuntimeError('Legacy repair loss has been removed. Use compute_repair_loss_from_batch().')

    def repair_refine_tokens(
            self,
            enc_out: torch.Tensor,
            init_tokens: torch.Tensor,
            use_repair: bool = True,
            steps: Optional[int] = None,
            conf_threshold: Optional[float] = None,
            n_vars: Optional[int] = None,
    ) -> torch.Tensor:
        """Compatibility wrapper around the new global edit-beam repair inference."""
        return self.repair_refine_tokens_edit_beam(
            enc_out=enc_out,
            init_tokens=init_tokens,
            use_repair=use_repair,
            steps=steps,
            conf_threshold=conf_threshold,
            editor_beam_k=1,
            revision_beam_k=1,
            n_vars=n_vars,
            trace_print=False,
        )

    # -------------------------------------------------------------------------
    # Repair inference (edit-space beam search)
    # -------------------------------------------------------------------------

    def _repair_is_need_closable(self, need: int, remaining_steps: int) -> bool:
        """A conservative feasibility check for prefix decoding.

        Each future token can reduce the deficit (need) by at most 1 (when choosing a leaf),
        so we require: need <= remaining_steps.
        We also keep at least one open slot if there are remaining steps.
        """
        need = int(need)
        remaining_steps = int(remaining_steps)
        if need <= 0:
            return False
        if remaining_steps < 0:
            return False
        # Must be able to close within remaining steps by spending leaves.
        return need <= remaining_steps

    def _repair_decode_rewrite_subtree_beam(
            self,
            *,
            orig_tokens: torch.Tensor,  # [1,L]
            skel_tokens: torch.Tensor,  # [1,L]
            enc_out: torch.Tensor,  # [1,Le,D]
            hole_positions: List[int],  # positions in [0,L)
            action_ids: Optional[torch.Tensor],  # [1,L] or None
            hole_mask: torch.Tensor,  # [1,L] bool
            rewrite_beam_size: int,
            allowed_leaf_ids: List[int],
    ) -> Tuple[Optional[List[int]], float]:
        """Decode one syntax-valid subtree with a small local beam.

        This local beam is the intra-action generator used inside a single REWRITE/INSERT edit.
        Ranking uses raw cumulative log-probability; no extra length-penalty hyperparameter is used.
        """
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None

        rewrite_beam_size = int(max(1, rewrite_beam_size))
        allowed_leaf_ids = [int(x) for x in allowed_leaf_ids]

        allowed_ops = [int(x) for x in helper.op_ids]
        allowed_pool = [v for v in (allowed_ops + allowed_leaf_ids) if helper.arity(v) >= 0]
        if not allowed_pool:
            return None, float('-inf')

        bos = int(getattr(self, 'repair_tag_insert_id', int(skel_tokens[0, hole_positions[0]].item()))) if hole_positions else None
        gen0 = skel_tokens.clone()
        if hole_positions and bos is not None:
            gen0[0, hole_positions[0]] = int(bos)

        # Beam state: (raw_logp, need, gen_tokens, produced)
        beam = [(0.0, 1, gen0, [])]
        best_complete = None  # (raw_logp, produced)

        for step_i, pos in enumerate(hole_positions):
            new_beam = []
            for logp_sum, need, gen_tokens, produced in beam:
                if need == 0:
                    new_beam.append((logp_sum, need, gen_tokens, produced))
                    continue

                token_logits = self.repair_generator_logits(
                    orig_tokens=orig_tokens,
                    edit_tokens=gen_tokens,
                    enc_out=enc_out,
                    action_ids=action_ids,
                    hole_mask=hole_mask,
                )[0, pos]
                token_logp = F.log_softmax(token_logits, dim=-1)

                remaining_slots = len(hole_positions) - (step_i + 1)
                feasible = []
                for v in allowed_pool:
                    a = helper.arity(v)
                    if a < 0:
                        continue
                    need2 = need - 1 + a
                    if need2 < 0:
                        continue
                    if remaining_slots == 0:
                        if need2 != 0:
                            continue
                    else:
                        if need2 != 0 and (not self._repair_is_need_closable(need2, remaining_slots)):
                            continue
                    feasible.append(v)

                if not feasible:
                    continue

                feas = torch.tensor(feasible, device=token_logp.device, dtype=torch.long)
                vals = token_logp.index_select(0, feas)
                topk = min(int(rewrite_beam_size), int(vals.numel()))
                top_vals, top_idx = torch.topk(vals, k=topk, largest=True)

                for v_logp, j in zip(top_vals.tolist(), top_idx.tolist()):
                    v = int(feasible[int(j)])
                    a = helper.arity(v)
                    need2 = int(need - 1 + a)

                    gen2 = gen_tokens.clone()
                    if step_i + 1 < len(hole_positions):
                        gen2[0, hole_positions[step_i + 1]] = v

                    prod2 = produced + [v]
                    lp2 = float(logp_sum + float(v_logp))
                    if need2 == 0:
                        if (best_complete is None) or (lp2 > best_complete[0]):
                            best_complete = (lp2, prod2)
                    new_beam.append((lp2, need2, gen2, prod2))

            if not new_beam:
                break
            new_beam.sort(key=lambda x: float(x[0]), reverse=True)
            beam = new_beam[:rewrite_beam_size]
            if all(st[1] == 0 for st in beam):
                break

        if best_complete is None:
            if not beam:
                return None, float('-inf')
            beam.sort(key=lambda x: float(x[0]), reverse=True)
            return None, float(beam[0][0])

        return list(best_complete[1]), float(best_complete[0])

    def _repair_apply_one_edit_scored(
            self,
            *,
            tokens: torch.Tensor,  # [1,L]
            enc_out: torch.Tensor,  # [1,Le,D]
            root: int,
            act: int,
            act_logp: float,
            allowed_leaf_ids: List[int],
            rewrite_beam_size: int,
            max_body_len: int,
    ) -> Tuple[Optional[torch.Tensor], float, int]:
        """Apply exactly one edit and return (new_tokens, total_logp, edit_size)."""
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None

        device = tokens.device
        toks = tokens[0].detach().cpu().tolist()
        body = helper.extract_body(toks)
        if (not body) or (not helper.validate_body(body)):
            return None, -1e9, 0
        L = int(tokens.shape[1])
        if not (0 <= int(root) < len(body)):
            return None, -1e9, 0

        root = int(root)
        act = int(act)
        end = int(helper.subtree_end(body, root))
        cur_tok = int(body[root])
        cur_arity = int(helper.arity(cur_tok))

        repl_tag_id = int(self.repair_tag_repl_id) if hasattr(self, 'repair_tag_repl_id') else None
        ins_tag_id = int(self.repair_tag_insert_id) if hasattr(self, 'repair_tag_insert_id') else None
        if repl_tag_id is None or ins_tag_id is None:
            return None, -1e9, 0

        gen_logp_sum = 0.0
        new_body: Optional[List[int]] = None
        edit_size: int = 0

        if int(act) == int(ACT_REPLACE):
            edit_size = 1
            if cur_arity == 0:
                cand_tokens = [int(x) for x in allowed_leaf_ids if int(x) != cur_tok]
            elif cur_arity == 1:
                cand_tokens = [int(x) for x in helper.unary_ids if int(x) != cur_tok]
            elif cur_arity == 2:
                cand_tokens = [int(x) for x in helper.binary_ids if int(x) != cur_tok]
            else:
                cand_tokens = []
            cand_tokens = [v for v in cand_tokens if int(helper.arity(int(v))) == int(cur_arity)]
            if not cand_tokens:
                return None, -1e9, 0

            skel_body = list(body)
            skel_body[root] = repl_tag_id
            skel_tokens = torch.tensor([helper.pack(skel_body, max_len=L)], device=device, dtype=torch.long)

            hole_mask = torch.zeros_like(skel_tokens, dtype=torch.bool)
            hole_mask[0, 1 + root] = True
            action_ids = torch.zeros_like(skel_tokens, dtype=torch.long)
            action_ids[0, 1 + root] = int(ACT_REPLACE)

            token_logits = self.repair_replace_delete_logits(
                orig_tokens=tokens,
                edit_tokens=skel_tokens,
                enc_out=enc_out,
                action_ids=action_ids,
                hole_mask=hole_mask,
                root_positions=torch.tensor([1 + root], device=device, dtype=torch.long),
                action_label=int(ACT_REPLACE),
            )[0]
            token_logp = F.log_softmax(token_logits, dim=-1)

            cand = torch.tensor(cand_tokens, device=device, dtype=torch.long)
            vals = token_logp.index_select(0, cand)
            j = int(torch.argmax(vals).item())
            new_tok = int(cand_tokens[j])
            gen_logp_sum = float(token_logp[new_tok].item())
            new_body = list(body)
            new_body[root] = new_tok

        elif int(act) == int(ACT_DELETE_SUBTREE):
            if cur_arity <= 0 or not allowed_leaf_ids:
                return None, -1e9, 0

            skel_body = list(body[:root]) + [repl_tag_id] + list(body[end:])
            if len(skel_body) > max_body_len:
                return None, -1e9, 0
            skel_tokens = torch.tensor([helper.pack(skel_body, max_len=L)], device=device, dtype=torch.long)

            hole_mask = torch.zeros_like(skel_tokens, dtype=torch.bool)
            hole_mask[0, 1 + root] = True
            action_ids = torch.zeros_like(skel_tokens, dtype=torch.long)
            action_ids[0, 1 + root] = int(ACT_DELETE_SUBTREE)

            token_logits = self.repair_replace_delete_logits(
                orig_tokens=tokens,
                edit_tokens=skel_tokens,
                enc_out=enc_out,
                action_ids=action_ids,
                hole_mask=hole_mask,
                root_positions=torch.tensor([1 + root], device=device, dtype=torch.long),
                action_label=int(ACT_DELETE_SUBTREE),
            )[0]
            token_logp = F.log_softmax(token_logits, dim=-1)

            leaf = torch.tensor([int(x) for x in allowed_leaf_ids], device=device, dtype=torch.long)
            vals = token_logp.index_select(0, leaf)
            j = int(torch.argmax(vals).item())
            leaf_tok = int(allowed_leaf_ids[j])
            gen_logp_sum = float(token_logp[leaf_tok].item())
            new_body = list(body[:root]) + [leaf_tok] + list(body[end:])

        elif int(act) in (int(ACT_REWRITE_SUBTREE), int(ACT_INSERT)):
            if int(act) == int(ACT_INSERT):
                if cur_arity != 0:
                    return None, -1e9, 0
                span_len = 1
            else:
                if cur_arity <= 0:
                    return None, -1e9, 0
                span_len = int(end - root)

            edit_size = int(span_len)
            slots = int(max(1, min(max_body_len, self.repair_fixed_hole_slots)))
            if slots <= 0:
                return None, -1e9, 0

            skel_body = list(body[:root]) + [ins_tag_id] * slots + list(body[end:])
            if len(skel_body) > max_body_len:
                return None, -1e9, 0
            skel_tokens = torch.tensor([helper.pack(skel_body, max_len=L)], device=device, dtype=torch.long)

            hole_positions = [1 + root + j for j in range(slots) if (1 + root + j) < L]
            hole_mask = torch.zeros_like(skel_tokens, dtype=torch.bool)
            for p in hole_positions:
                hole_mask[0, p] = True

            action_ids = torch.zeros_like(skel_tokens, dtype=torch.long)
            for p in hole_positions:
                action_ids[0, p] = int(act)

            subtree, lp_sub = self._repair_decode_rewrite_subtree_beam(
                orig_tokens=tokens,
                skel_tokens=skel_tokens,
                enc_out=enc_out,
                hole_positions=hole_positions,
                action_ids=action_ids,
                hole_mask=hole_mask,
                rewrite_beam_size=rewrite_beam_size,
                allowed_leaf_ids=allowed_leaf_ids,
            )
            if subtree is None:
                return None, -1e9, 0
            subtree = [int(x) for x in subtree]
            root_arity = int(helper.arity(int(subtree[0]))) if subtree else -1
            if int(act) == int(ACT_INSERT):
                if len(subtree) <= 1 or root_arity <= 0:
                    return None, -1e9, 0
            else:
                if root_arity <= 0:
                    return None, -1e9, 0

            gen_logp_sum = float(lp_sub)
            new_body = list(body[:root]) + list(subtree) + list(body[end:])

        else:
            return None, -1e9, 0

        if new_body is None or len(new_body) > max_body_len or (not helper.validate_body(new_body)):
            return None, -1e9, 0
        new_tokens = torch.tensor([helper.pack(new_body, max_len=L)], device=device, dtype=torch.long)
        new_tokens = self._truncate_after_F(new_tokens)
        total_lp = float(act_logp + gen_logp_sum)
        return new_tokens, total_lp, int(edit_size)

    def _removed_repair_candidate_pool_2d_beam(
            self,
            *args,
            **kwargs,
    ):
        raise RuntimeError('Legacy repair candidate beam has been removed. Use repair_candidate_pool_2d_beam().')

    def _removed_repair_refine_tokens_edit_beam(
            self,
            *args,
            **kwargs,
    ):
        raise RuntimeError('Legacy repair inference has been removed. Use repair_refine_tokens_edit_beam().')

    def _removed_repair_refine_tokens_edit_beam_v2(
            self,
            *args,
            **kwargs,
    ):
        raise RuntimeError('Legacy repair inference has been removed. Use repair_refine_tokens_edit_beam().')

    def _repair_frontier_action_cost(self, act: int, cur_sub: List[int], gt_sub: List[int]) -> float:
        lam = float(self.repair_frontier_lambda)
        cur_n = int(max(1, len(cur_sub)))
        gt_n = int(max(1, len(gt_sub)))
        if int(act) == int(ACT_REPLACE):
            return 1.0
        if int(act) == int(ACT_DELETE_SUBTREE):
            return 1.0 + lam * float(max(0, cur_n ))
        if int(act) == int(ACT_INSERT):
            return 1.0 + lam * float(max(0, gt_n ))
        if int(act) == int(ACT_REWRITE_SUBTREE):
            return 1.0 + lam * float(  max(0, gt_n))
        return 0.0

    def _oracle_parallel_frontier_to_gt(
            self,
            *,
            cur_body: List[int],
            gt_body: List[int],
            helper: PrefixRepairHelper,
            max_body_len: int,
    ) -> List[Dict[str, Any]]:
        cur_body = [int(x) for x in list(cur_body)[:max_body_len]]
        gt_body = [int(x) for x in list(gt_body)[:max_body_len]]
        if (not cur_body) or (not gt_body):
            return []
        if (not helper.validate_body(cur_body)) or (not helper.validate_body(gt_body)):
            return []
        if cur_body == gt_body:
            return []

        memo: Dict[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]], Tuple[float, int, List[Dict[str, Any]]]] = {}
        INF = float('inf')
        rewrite_cap = int(max(2, self.repair_direct_rewrite_max_nodes))

        def _rank_key(item: Tuple[float, int, List[Dict[str, Any]]]):
            cost, kind_rank, plan = item
            avg_depth = (sum(len(e['path']) for e in plan) / max(1, len(plan))) if plan else 0.0
            return (float(cost), int(kind_rank), int(len(plan)), float(avg_depth))

        def solve(cur_sub: List[int], gt_sub: List[int], path: Tuple[int, ...]):
            key = (tuple(cur_sub), tuple(gt_sub), tuple(path))
            if key in memo:
                return memo[key]
            if cur_sub == gt_sub:
                memo[key] = (0.0, 1, [])
                return memo[key]

            candidates: List[Tuple[float, int, List[Dict[str, Any]]]] = []

            act, target_subtree, prev_root_token = self._classify_local_repair_action(
                cur_sub=list(cur_sub),
                prev_sub=list(gt_sub),
                helper=helper,
            )
            allow_local = int(act) != int(ACT_KEEP)
            if int(act) == int(ACT_REWRITE_SUBTREE):
                allow_local = allow_local and int(len(gt_sub)) <= rewrite_cap
            if allow_local:
                c_local = self._repair_frontier_action_cost(int(act), list(cur_sub), list(gt_sub))
                candidates.append((
                    float(c_local),
                    1,
                    [{
                        'path': tuple(path),
                        'action': int(act),
                        'target_subtree': list(target_subtree) if target_subtree is not None else None,
                        'prev_root_token': int(prev_root_token) if prev_root_token is not None else None,
                    }]
                ))

            cur_root = int(cur_sub[0]) if cur_sub else -1
            gt_root = int(gt_sub[0]) if gt_sub else -1
            a_cur = int(helper.arity(cur_root)) if cur_sub else -1
            a_gt = int(helper.arity(gt_root)) if gt_sub else -1
            if cur_root == gt_root and a_cur == a_gt and a_cur > 0:
                cur_children = self._oracle_split_children(tuple(cur_sub), helper)
                gt_children = self._oracle_split_children(tuple(gt_sub), helper)
                if len(cur_children) == len(gt_children):
                    split_cost = 0.0
                    split_plan: List[Dict[str, Any]] = []
                    ok = True
                    for k, (cch, gch) in enumerate(zip(cur_children, gt_children)):
                        ck, _, pk = solve(list(cch), list(gch), tuple(path) + (int(k),))
                        if math.isinf(ck):
                            ok = False
                            break
                        split_cost += float(ck)
                        split_plan.extend(pk)
                    if ok:
                        candidates.append((float(split_cost), 0, split_plan))

            if not candidates:
                memo[key] = (INF, 1, [])
                return memo[key]

            best = min(candidates, key=_rank_key)
            memo[key] = best
            return best

        total_cost, _, rel_plan = solve(cur_body, gt_body, ())
        if math.isinf(total_cost):
            return []

        amap = self._parse_prefix_addr_map(cur_body, helper)
        frontier: List[Dict[str, Any]] = []
        for e in rel_plan:
            node = amap.get(tuple(e['path']), None)
            if node is None:
                return []
            s = int(node['start'])
            t = int(node['end'])
            frontier.append({
                'path': tuple(e['path']),
                'root_idx': int(s),
                'action': int(e['action']),
                'target_subtree': list(e['target_subtree']) if e.get('target_subtree', None) is not None else None,
                'prev_root_token': int(e['prev_root_token']) if e.get('prev_root_token', None) is not None else None,
                'span_start': int(s),
                'span_end': int(t),
            })
        frontier.sort(key=lambda z: int(z['root_idx']))
        return frontier

    def _repair_build_global_skeleton(
            self,
            *,
            cur_body: List[int],
            edits: List[Dict[str, Any]],
            helper: PrefixRepairHelper,
            max_body_len: int,
            include_targets: bool,
    ) -> Optional[Dict[str, Any]]:
        cur_body = [int(x) for x in cur_body[:max_body_len]]
        edits = sorted(list(edits), key=lambda z: int(z['span_start']))
        if not helper.validate_body(cur_body):
            return None
        repl_tag_id = int(self.repair_tag_repl_id) if self.repair_tag_repl_id is not None else int(helper._random_const_leaf())
        insert_tag_id = int(self.repair_tag_insert_id) if self.repair_tag_insert_id is not None else int(helper._random_const_leaf())
        skel_body: List[int] = []
        blocks: List[Dict[str, Any]] = []
        cursor = 0
        for e in edits:
            s = int(e['span_start'])
            t = int(e['span_end'])
            act = int(e['action'])
            if s < cursor:
                return None
            skel_body.extend(cur_body[cursor:s])
            body_start = int(len(skel_body))
            cur_root_tok = int(cur_body[s])
            cur_root_arity = int(helper.arity(cur_root_tok))
            if act in (ACT_REPLACE, ACT_DELETE_SUBTREE):
                skel_body.append(int(repl_tag_id))
                block = {
                    'action': int(act),
                    'body_start': int(body_start),
                    'slots': 1,
                    'hole_token_id': int(repl_tag_id),
                    'target_tokens': ([int(e['prev_root_token'])] if include_targets and e.get('prev_root_token', None) is not None else []),
                    'current_root_token': int(cur_root_tok),
                    'current_root_arity': int(cur_root_arity),
                    'span_start': int(s),
                    'span_end': int(t),
                }
            else:
                slots = int(max(1, self.repair_fixed_hole_slots))
                skel_body.extend([int(insert_tag_id)] * int(slots))
                block = {
                    'action': int(act),
                    'body_start': int(body_start),
                    'slots': int(slots),
                    'hole_token_id': int(insert_tag_id),
                    'target_tokens': ([int(x) for x in (e.get('target_subtree', []) or [])] if include_targets else []),
                    'current_root_token': int(cur_root_tok),
                    'current_root_arity': int(cur_root_arity),
                    'span_start': int(s),
                    'span_end': int(t),
                }
            blocks.append(block)
            cursor = int(t)
        skel_body.extend(cur_body[cursor:])
        if len(skel_body) > int(max_body_len):
            return None
        return {'skel_body': skel_body, 'blocks': blocks}

    def _repair_prepare_editor_io(
            self,
            *,
            skel_body: List[int],
            blocks: List[Dict[str, Any]],
            helper: PrefixRepairHelper,
            L: int,
    ) -> Dict[str, Any]:
        editor_inp = list(helper.pack(skel_body, max_len=L))
        hole_mask = [False] * L
        action_ids = [ACT_KEEP] * L
        gen_tgt = [-100] * L
        blocks_out: List[Dict[str, Any]] = []
        for block_idx, block in enumerate(blocks):
            body_start = int(block['body_start'])
            slots = int(block['slots'])
            pos_list = []
            for k in range(slots):
                pos = 1 + body_start + k
                if pos < L:
                    pos_list.append(int(pos))
                    hole_mask[pos] = True
                    action_ids[pos] = int(block['action'])
            tgt = [int(x) for x in block.get('target_tokens', [])]
            tag_id = int(block['hole_token_id'])
            for j, pos in enumerate(pos_list):
                if j == 0:
                    editor_inp[pos] = int(tag_id)
                elif j - 1 < len(tgt):
                    editor_inp[pos] = int(tgt[j - 1])
                else:
                    editor_inp[pos] = int(tag_id)
                if j < len(tgt):
                    gen_tgt[pos] = int(tgt[j])
            blk = dict(block)
            blk['hole_positions'] = [int(x) for x in pos_list]
            blk['block_idx'] = int(block_idx)
            blocks_out.append(blk)
        return {
            'editor_inp': editor_inp,
            'hole_mask': hole_mask,
            'action_ids': action_ids,
            'gen_tgt': gen_tgt,
            'blocks': blocks_out,
        }

    def _repair_select_single_edit_candidates(
            self,
            *,
            body: List[int],
            action_logits_body: torch.Tensor,
            helper: PrefixRepairHelper,
            conf_threshold: float,
            top_k: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        if action_logits_body.numel() == 0:
            return [], 0.0
        act_prob = torch.softmax(action_logits_body, dim=-1)
        best_nonkeep_prob = float(act_prob[:, 1:].max().item()) if action_logits_body.numel() > 0 else 0.0
        candidates: List[Dict[str, Any]] = []
        top_k = int(max(1, top_k))
        for i in range(int(action_logits_body.shape[0])):
            cur_tok = int(body[i])
            cur_arity = int(helper.arity(cur_tok))
            s = int(i)
            e = int(helper.subtree_end(body, i))
            for j in range(1, int(action_logits_body.shape[1])):
                act = int(TAGGER_ACTIONS[j])
                p = float(act_prob[i, j].item())
                if p < float(conf_threshold):
                    continue
                if act == ACT_INSERT and cur_arity != 0:
                    continue
                if act in (ACT_DELETE_SUBTREE, ACT_REWRITE_SUBTREE) and cur_arity <= 0:
                    continue
                if act == ACT_REPLACE and cur_arity < 0:
                    continue
                candidates.append({
                    'root_idx': int(i),
                    'action': int(act),
                    'score': float(action_logits_body[i, j].item()),
                    'logp': float(math.log(max(p, 1e-12))),
                    'prob': float(p),
                    'span_start': int(s),
                    'span_end': int(e),
                })
        candidates.sort(key=lambda z: (float(z['score']), float(z['logp'])), reverse=True)
        return [dict(c) for c in candidates[:top_k]], float(best_nonkeep_prob)

    def _repair_compose_body_from_skeleton(
            self,
            *,
            skel_body: List[int],
            blocks: List[Dict[str, Any]],
            produced_by_block: Dict[int, List[int]],
    ) -> List[int]:
        out: List[int] = []
        cursor = 0
        for block in sorted(blocks, key=lambda z: int(z['body_start'])):
            bs = int(block['body_start'])
            slots = int(block['slots'])
            out.extend(skel_body[cursor:bs])
            out.extend([int(x) for x in produced_by_block.get(int(block['block_idx']), [])])
            cursor = int(bs + slots)
        out.extend(skel_body[cursor:])
        return [int(x) for x in out]

    def _repair_decode_block_candidates(
            self,
            *,
            orig_tokens: torch.Tensor,
            editor_tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: Optional[torch.Tensor],
            hole_mask: torch.Tensor,
            block: Dict[str, Any],
            allowed_leaf_ids: List[int],
            beam_size: int,
    ) -> List[Dict[str, Any]]:
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None
        beam_size = int(max(1, beam_size))
        act = int(block['action'])
        hole_positions = [int(x) for x in block.get('hole_positions', [])]
        if not hole_positions:
            return []
        device = editor_tokens.device

        if act == ACT_REPLACE:
            cur_tok = int(block['current_root_token'])
            cur_arity = int(block['current_root_arity'])
            if cur_arity == 0:
                cand_tokens = [int(x) for x in allowed_leaf_ids if int(x) != cur_tok]
            elif cur_arity == 1:
                cand_tokens = [int(x) for x in helper.unary_ids if int(x) != cur_tok]
            elif cur_arity == 2:
                cand_tokens = [int(x) for x in helper.binary_ids if int(x) != cur_tok]
            else:
                cand_tokens = []
            if not cand_tokens:
                return []
            logits = self.repair_replace_delete_logits(
                orig_tokens=orig_tokens,
                edit_tokens=editor_tokens,
                enc_out=enc_out,
                action_ids=action_ids,
                hole_mask=hole_mask,
                root_positions=torch.tensor([hole_positions[0]], device=device, dtype=torch.long),
                action_label=int(ACT_REPLACE),
            )[0]
            logp = F.log_softmax(logits, dim=-1)
            cand = torch.tensor(cand_tokens, device=device, dtype=torch.long)
            vals = logp.index_select(0, cand)
            k = int(min(beam_size, vals.numel()))
            top_vals, top_idx = torch.topk(vals, k=k, largest=True)
            out = []
            for v_lp, j in zip(top_vals.tolist(), top_idx.tolist()):
                tok = int(cand_tokens[int(j)])
                out.append({'tokens': editor_tokens.clone(), 'produced': [int(tok)], 'raw_logp': float(v_lp)})
            return out

        if act == ACT_DELETE_SUBTREE:
            if not allowed_leaf_ids:
                return []
            logits = self.repair_replace_delete_logits(
                orig_tokens=orig_tokens,
                edit_tokens=editor_tokens,
                enc_out=enc_out,
                action_ids=action_ids,
                hole_mask=hole_mask,
                root_positions=torch.tensor([hole_positions[0]], device=device, dtype=torch.long),
                action_label=int(ACT_DELETE_SUBTREE),
            )[0]
            logp = F.log_softmax(logits, dim=-1)
            cand = torch.tensor([int(x) for x in allowed_leaf_ids], device=device, dtype=torch.long)
            vals = logp.index_select(0, cand)
            k = int(min(beam_size, vals.numel()))
            top_vals, top_idx = torch.topk(vals, k=k, largest=True)
            out = []
            for v_lp, j in zip(top_vals.tolist(), top_idx.tolist()):
                tok = int(allowed_leaf_ids[int(j)])
                out.append({'tokens': editor_tokens.clone(), 'produced': [int(tok)], 'raw_logp': float(v_lp)})
            return out

        # INSERT / REWRITE: decode one closed subtree within the block slots.
        allowed_pool = [int(x) for x in helper.op_ids] + [int(x) for x in allowed_leaf_ids]
        allowed_pool = [int(x) for x in allowed_pool if int(helper.arity(int(x))) >= 0]
        if not allowed_pool:
            return []

        beam = [{'tokens': editor_tokens.clone(), 'raw_logp': 0.0, 'need': 1, 'produced': []}]
        complete: List[Dict[str, Any]] = []
        for step_i, pos in enumerate(hole_positions):
            new_beam: List[Dict[str, Any]] = []
            for st in beam:
                need = int(st['need'])
                if need == 0:
                    complete.append({'tokens': st['tokens'], 'produced': list(st['produced']), 'raw_logp': float(st['raw_logp'])})
                    continue
                logits = self.repair_generator_logits(
                    orig_tokens=orig_tokens,
                    edit_tokens=st['tokens'],
                    enc_out=enc_out,
                    action_ids=action_ids,
                    hole_mask=hole_mask,
                )[0, pos]
                logp = F.log_softmax(logits, dim=-1)
                remaining_slots = int(len(hole_positions) - step_i - 1)
                feasible: List[int] = []
                for v in allowed_pool:
                    a = int(helper.arity(int(v)))
                    if step_i == 0 and a <= 0:
                        continue
                    need2 = int(need - 1 + a)
                    if need2 < 0:
                        continue
                    if remaining_slots == 0:
                        if need2 != 0:
                            continue
                    else:
                        if need2 != 0 and (not self._repair_is_need_closable(need2, remaining_slots)):
                            continue
                    feasible.append(int(v))
                if not feasible:
                    continue
                cand = torch.tensor(feasible, device=device, dtype=torch.long)
                vals = logp.index_select(0, cand)
                k = int(min(beam_size, vals.numel()))
                top_vals, top_idx = torch.topk(vals, k=k, largest=True)
                for v_lp, j in zip(top_vals.tolist(), top_idx.tolist()):
                    tok = int(feasible[int(j)])
                    a = int(helper.arity(tok))
                    need2 = int(need - 1 + a)
                    toks2 = st['tokens'].clone()
                    if step_i + 1 < len(hole_positions):
                        toks2[0, hole_positions[step_i + 1]] = int(tok)
                    prod2 = list(st['produced']) + [int(tok)]
                    st2 = {'tokens': toks2, 'raw_logp': float(st['raw_logp'] + float(v_lp)), 'need': int(need2), 'produced': prod2}
                    if need2 == 0:
                        complete.append({'tokens': toks2, 'produced': prod2, 'raw_logp': float(st2['raw_logp'])})
                    else:
                        new_beam.append(st2)
            if not new_beam and complete:
                break
            new_beam.sort(key=lambda z: float(z['raw_logp']), reverse=True)
            beam = new_beam[:beam_size]
            if all(int(st['need']) == 0 for st in beam):
                break
        if not complete:
            complete = [{'tokens': st['tokens'], 'produced': list(st['produced']), 'raw_logp': float(st['raw_logp'])} for st in beam if int(st.get('need', 1)) == 0]
        complete = [c for c in complete if len(c.get('produced', [])) > 0]
        complete.sort(key=lambda z: float(z['raw_logp']), reverse=True)
        return complete[:beam_size]

    def _repair_decode_global_skeleton_beam(
            self,
            *,
            orig_tokens: torch.Tensor,
            skel_body: List[int],
            blocks: List[Dict[str, Any]],
            editor_inp_tokens: torch.Tensor,
            enc_out: torch.Tensor,
            action_ids: Optional[torch.Tensor],
            hole_mask: torch.Tensor,
            allowed_leaf_ids: List[int],
            beam_size: int,
            max_body_len: int,
    ) -> List[Dict[str, Any]]:
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None
        beam_size = int(max(1, beam_size))
        if not blocks:
            body = [int(x) for x in skel_body]
            if helper.validate_body(body):
                tok = torch.tensor([helper.pack(body, max_len=editor_inp_tokens.shape[1])], device=editor_inp_tokens.device, dtype=torch.long)
                return [{'tokens': tok, 'body': body, 'raw_logp': 0.0}]
            return []

        beam = [{'tokens': editor_inp_tokens.clone(), 'raw_logp': 0.0, 'produced_by_block': {}}]
        for block in blocks:
            expanded: List[Dict[str, Any]] = []
            for st in beam:
                cand_blocks = self._repair_decode_block_candidates(
                    orig_tokens=orig_tokens,
                    editor_tokens=st['tokens'],
                    enc_out=enc_out,
                    action_ids=action_ids,
                    hole_mask=hole_mask,
                    block=block,
                    allowed_leaf_ids=allowed_leaf_ids,
                    beam_size=beam_size,
                )
                for cand in cand_blocks:
                    produced_map = dict(st['produced_by_block'])
                    produced_map[int(block['block_idx'])] = [int(x) for x in cand['produced']]
                    expanded.append({
                        'tokens': cand['tokens'],
                        'raw_logp': float(st['raw_logp'] + cand['raw_logp']),
                        'produced_by_block': produced_map,
                    })
            if not expanded:
                return []
            expanded.sort(key=lambda z: float(z['raw_logp']), reverse=True)
            beam = expanded[:beam_size]

        out: List[Dict[str, Any]] = []
        seen: Set[Tuple[int, ...]] = set()
        L = int(editor_inp_tokens.shape[1])
        for st in beam:
            body = self._repair_compose_body_from_skeleton(
                skel_body=skel_body,
                blocks=blocks,
                produced_by_block=st['produced_by_block'],
            )
            if (not body) or len(body) > int(max_body_len) or (not helper.validate_body(body)):
                continue
            key = tuple(int(x) for x in body)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                'tokens': torch.tensor([helper.pack(body, max_len=L)], device=editor_inp_tokens.device, dtype=torch.long),
                'body': body,
                'raw_logp': float(st['raw_logp']),
            })
        out.sort(key=lambda z: float(z['raw_logp']), reverse=True)
        return out[:beam_size]

    def _repair_one_step_global(
            self,
            *,
            enc_out: torch.Tensor,
            init_tokens: torch.Tensor,
            conf_threshold: float,
            editor_beam_k: int,
            tagger_topk: int = 1,
            n_vars: Optional[int] = None,
            trace_print: bool = False,
    ) -> Dict[str, Any]:
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None
        tokens = self._truncate_after_F(init_tokens[:1].long())
        L = int(tokens.shape[1])
        max_body_len = max(1, L - 2)
        allowed_leaf_ids = helper.allowed_leaf_ids_from_nvars(int(n_vars) if n_vars is not None else 0)
        body = helper.extract_body(tokens[0].detach().cpu().tolist())
        if (not body) or (not helper.validate_body(body)):
            return {'successors': [], 'reason': 'invalid_body', 'best_nonkeep': 0.0, 'selected_edit': None}

        action_logits = self.repair_tagger_logits(tokens=tokens, enc_out=enc_out[:1])
        act_logits_body = action_logits[0, 1:1 + len(body), :]
        selected_edits, best_nonkeep = self._repair_select_single_edit_candidates(
            body=body,
            action_logits_body=act_logits_body,
            helper=helper,
            conf_threshold=float(conf_threshold),
            top_k=int(tagger_topk),
        )
        if not selected_edits:
            return {'successors': [], 'reason': 'confidence_below_threshold', 'best_nonkeep': float(best_nonkeep), 'selected_edit': None}

        successors = []
        best_selected_edit = None
        for edit in selected_edits:
            if best_selected_edit is None:
                best_selected_edit = dict(edit)
            tag_logp = float(edit['logp'])
            skel = self._repair_build_global_skeleton(
                cur_body=body,
                edits=[edit],
                helper=helper,
                max_body_len=max_body_len,
                include_targets=False,
            )
            if skel is None:
                continue
            editor_io = self._repair_prepare_editor_io(
                skel_body=skel['skel_body'],
                blocks=skel['blocks'],
                helper=helper,
                L=L,
            )
            editor_tokens = torch.tensor(editor_io['editor_inp'], device=tokens.device, dtype=torch.long).unsqueeze(0)
            hole_mask = torch.tensor(editor_io['hole_mask'], device=tokens.device, dtype=torch.bool).unsqueeze(0)
            action_ids = torch.tensor(editor_io['action_ids'], device=tokens.device, dtype=torch.long).unsqueeze(0)
            decoded = self._repair_decode_global_skeleton_beam(
                orig_tokens=tokens,
                skel_body=skel['skel_body'],
                blocks=editor_io['blocks'],
                editor_inp_tokens=editor_tokens,
                enc_out=enc_out[:1],
                action_ids=action_ids,
                hole_mask=hole_mask,
                allowed_leaf_ids=allowed_leaf_ids,
                beam_size=int(editor_beam_k),
                max_body_len=max_body_len,
            )
            for cand in decoded:
                raw_lp = float(tag_logp + cand['raw_logp'])
                successors.append({
                    'tokens': cand['tokens'],
                    'raw_logp': raw_lp,
                    'tag_logp': float(tag_logp),
                    'editor_logp': float(cand['raw_logp']),
                    'selected_edit': dict(edit),
                    'best_nonkeep': float(best_nonkeep),
                })
        successors.sort(key=lambda z: float(z['raw_logp']), reverse=True)
        reason = None if successors else 'skeleton_build_failed'
        return {'successors': successors, 'reason': reason, 'best_nonkeep': float(best_nonkeep), 'selected_edit': best_selected_edit}

    def compute_repair_loss_from_batch(
            self,
            batch,
            enc_out: torch.Tensor,
            output_logits: Optional[torch.Tensor] = None,
            batch_idx: Optional[int] = None,
    ):
        """Train Tagger and Editor from sampled chain states.

        For each sample we first construct one or more syntax-valid source states, build the full
        rooted edit chain from that source to the target, sample one chain index, and supervise only
        the residual frontier and the corresponding global-skeleton editor targets from that sampled
        chain state.
        """
        if not self.repair_enable:
            return None
        trg = batch[1].long()
        trg = self._truncate_after_F(trg)
        B, L = trg.shape

        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        if helper is None:
            return None
        max_body_len = max(1, L - 2)
        n_vars = int(getattr(self.cfg, 'num_features', 0))

        gt_bodies: List[List[int]] = []
        for row in trg.detach().cpu().tolist():
            gt_body = helper.extract_body(row)
            if (not gt_body) or (not helper.validate_body(gt_body)):
                gt_body = [int(helper._random_const_leaf())]
            gt_bodies.append([int(x) for x in gt_body[:max_body_len]])

        ar_init_bodies: Dict[int, List[int]] = {}
        if self.repair_source_use_ar:
            allowed_batch = [helper.allowed_leaf_ids_from_nvars(n_vars) for _ in range(B)]
            with torch.no_grad():
                init_tokens = self._ar_greedy_free_run_init(
                    enc_out=enc_out,
                    seq_len=L,
                    allowed_leaf_ids_batch=allowed_batch,
                    return_step_logits=False,
                )
            for b, row in enumerate(init_tokens.detach().cpu().tolist()):
                body = helper.extract_body(row)
                if (not body) or (not helper.validate_body(body)) or len(body) > max_body_len:
                    body = [int(helper._random_const_leaf())]
                ar_init_bodies[int(b)] = [int(x) for x in body[:max_body_len]]

        state_cur_tokens: List[List[int]] = []
        state_action_tgt: List[List[int]] = []
        state_editor_orig_inp: List[List[int]] = []
        state_editor_inp: List[List[int]] = []
        state_hole_mask: List[List[bool]] = []
        state_action_ids: List[List[int]] = []
        state_gen_tgt: List[List[int]] = []
        state_single_tgt: List[int] = []
        state_root_pos: List[int] = []
        state_action_label: List[int] = []
        state_enc_src_idx: List[int] = []
        state_body_len: List[int] = []
        state_has_editor: List[bool] = []
        trace_candidates: List[Dict[str, Any]] = []

        ar_state_count = 0
        synth_state_count = 0
        rollout_state_count = 0
        keep_self_state_count = 0

        use_full_chain_supervision = True

        def _append_single_edit_state(
                *,
                src_b: int,
                cur_body: List[int],
                edit: Dict[str, Any],
        ) -> bool:
            cur_body = [int(x) for x in cur_body[:max_body_len]]
            if (not cur_body) or (not helper.validate_body(cur_body)):
                return False
            root_idx = int(edit.get('root_idx', -1))
            action = int(edit.get('action', ACT_KEEP))
            if not (0 <= root_idx < len(cur_body)) or action not in TAGGER_ACTIONS or int(action) == int(ACT_KEEP):
                return False
            span_start = int(edit.get('span_start', root_idx))
            span_end = int(edit.get('span_end', helper.subtree_end(cur_body, root_idx)))
            if span_start < 0 or span_end <= span_start or span_end > len(cur_body):
                span_start = int(root_idx)
                span_end = int(helper.subtree_end(cur_body, root_idx))
            single_edit = {
                'root_idx': int(root_idx),
                'action': int(action),
                'target_subtree': [int(x) for x in (edit.get('target_subtree', []) or [])],
                'prev_root_token': int(edit['prev_root_token']) if edit.get('prev_root_token', None) is not None else None,
                'span_start': int(span_start),
                'span_end': int(span_end),
            }
            action_tgt = [int(ACT_KEEP) for _ in range(L)]
            action_tgt[1 + root_idx] = int(action)
            skel = self._repair_build_global_skeleton(
                cur_body=cur_body,
                edits=[single_edit],
                helper=helper,
                max_body_len=max_body_len,
                include_targets=True,
            )
            if skel is None:
                return False
            editor_io = self._repair_prepare_editor_io(
                skel_body=skel['skel_body'],
                blocks=skel['blocks'],
                helper=helper,
                L=L,
            )
            packed_cur = helper.pack(cur_body, max_len=L)
            single_tgt = -100
            if int(action) == int(ACT_REPLACE) and single_edit.get('prev_root_token', None) is not None:
                single_tgt = int(single_edit['prev_root_token'])
            elif int(action) == int(ACT_DELETE_SUBTREE) and single_edit.get('prev_root_token', None) is not None:
                single_tgt = int(single_edit['prev_root_token'])
            state_cur_tokens.append(packed_cur)
            state_action_tgt.append(action_tgt)
            state_editor_orig_inp.append(packed_cur)
            state_editor_inp.append(editor_io['editor_inp'])
            state_hole_mask.append(editor_io['hole_mask'])
            state_action_ids.append(editor_io['action_ids'])
            state_gen_tgt.append(editor_io['gen_tgt'])
            state_single_tgt.append(int(single_tgt))
            state_root_pos.append(int(1 + root_idx))
            state_action_label.append(int(action))
            state_enc_src_idx.append(int(src_b))
            state_body_len.append(int(len(cur_body)))
            state_has_editor.append(bool(any(editor_io['hole_mask'])) or int(action) in (int(ACT_REPLACE), int(ACT_DELETE_SUBTREE)))
            return True

        def append_training_state(src_b: int, source_body: List[int], source_type: str, source_meta: Optional[Dict[str, Any]] = None) -> int:
            source_body = [int(x) for x in source_body[:max_body_len]]
            if (not source_body) or (not helper.validate_body(source_body)):
                return 0
            source_meta = dict(source_meta or {})
            chain = self._oracle_build_chain(
                cur_body=source_body,
                gt_body=gt_bodies[src_b],
                helper=helper,
                max_body_len=max_body_len,
            )

            added = 0
            # Single-edit full-chain supervision only. The legacy parallel-frontier branch has been removed.
            for step_idx, rec in enumerate(chain):
                    cur_body = [int(x) for x in rec.get('cur_body', [])[:max_body_len]]
                    ok = _append_single_edit_state(
                        src_b=int(src_b),
                        cur_body=cur_body,
                        edit={
                            'root_idx': int(rec.get('root_idx', -1)),
                            'action': int(rec.get('rev_action', ACT_KEEP)),
                            'target_subtree': [int(x) for x in (rec.get('target_subtree', []) or [])],
                            'prev_root_token': int(rec['prev_root_token']) if rec.get('prev_root_token', None) is not None else None,
                            'span_start': int(rec.get('span_start', rec.get('root_idx', -1))),
                            'span_end': int(rec.get('span_end', -1)),
                        },
                    )
                    if ok:
                        added += 1
            if self._repair_trace_is_enabled():
                trace_candidates.append({
                    'trace_id': f"{('train' if self.training else 'val')}-e{int(self.current_epoch)}-gs{int(self.global_step)}-b{int(batch_idx if batch_idx is not None else -1)}-src{int(src_b)}-{source_type}-{len(trace_candidates)}",
                    'split': 'train' if self.training else 'val',
                    'epoch': int(self.current_epoch),
                    'global_step': int(self.global_step),
                    'batch_idx': int(batch_idx if batch_idx is not None else -1),
                    'src_b': int(src_b),
                    'source_type': str(source_type),
                    'gt_body': [int(x) for x in gt_bodies[src_b]],
                    'source_body': [int(x) for x in source_body],
                    'sampled_cur_body': [int(x) for x in source_body],
                    'chain_sampling_distribution': 'all_steps_in_order',
                    'chain_sampled_index': '',
                    'chain_length': int(len(chain)),
                    'oracle_chain': [
                        {
                            'cur_body': [int(x) for x in rec.get('cur_body', [])],
                            'prev_body': [int(x) for x in rec.get('prev_body', [])],
                            'rev_action': int(rec.get('rev_action', ACT_KEEP)),
                            'root_idx': int(rec.get('root_idx', -1)),
                            'span_start': int(rec.get('span_start', -1)),
                            'span_end': int(rec.get('span_end', -1)),
                            'target_subtree': [int(x) for x in (rec.get('target_subtree', []) or [])],
                            'prev_root_token': int(rec['prev_root_token']) if rec.get('prev_root_token', None) is not None else None,
                            'edit_content': [int(x) for x in (rec.get('edit_content', []) or [])],
                        }
                        for rec in chain
                    ],
                    'frontier': [],
                    'source_meta': {**source_meta, 'supervision_mode': self.repair_supervision_mode, 'training_states_added': int(added)},
                })
            return int(added)

        def append_keep_state(src_b: int, body: List[int], source_type: str = 'gt_keep', source_meta: Optional[Dict[str, Any]] = None) -> int:
            body = [int(x) for x in body[:max_body_len]]
            if (not body) or (not helper.validate_body(body)):
                return 0
            action_tgt = [int(ACT_KEEP) for _ in range(L)]
            skel = self._repair_build_global_skeleton(
                cur_body=body,
                edits=[],
                helper=helper,
                max_body_len=max_body_len,
                include_targets=True,
            )
            if skel is None:
                return 0
            editor_io = self._repair_prepare_editor_io(
                skel_body=skel['skel_body'],
                blocks=skel['blocks'],
                helper=helper,
                L=L,
            )
            packed_body = helper.pack(body, max_len=L)
            state_cur_tokens.append(packed_body)
            state_action_tgt.append(action_tgt)
            state_editor_orig_inp.append(packed_body)
            state_editor_inp.append(editor_io['editor_inp'])
            state_hole_mask.append(editor_io['hole_mask'])
            state_action_ids.append(editor_io['action_ids'])
            state_gen_tgt.append(editor_io['gen_tgt'])
            state_single_tgt.append(-100)
            state_root_pos.append(0)
            state_action_label.append(int(ACT_KEEP))
            state_enc_src_idx.append(int(src_b))
            state_body_len.append(int(len(body)))
            state_has_editor.append(False)
            if self._repair_trace_is_enabled():
                meta = dict(source_meta or {})
                meta['supervision_mode'] = f"{self.repair_supervision_mode}+keep_self"
                trace_candidates.append({
                    'trace_id': f"{('train' if self.training else 'val')}-e{int(self.current_epoch)}-gs{int(self.global_step)}-b{int(batch_idx if batch_idx is not None else -1)}-src{int(src_b)}-{source_type}-{len(trace_candidates)}",
                    'split': 'train' if self.training else 'val',
                    'epoch': int(self.current_epoch),
                    'global_step': int(self.global_step),
                    'batch_idx': int(batch_idx if batch_idx is not None else -1),
                    'src_b': int(src_b),
                    'source_type': str(source_type),
                    'gt_body': [int(x) for x in gt_bodies[src_b]],
                    'source_body': [int(x) for x in body],
                    'sampled_cur_body': [int(x) for x in body],
                    'chain_sampling_distribution': 'point_mass_gt_keep',
                    'chain_sampled_index': '',
                    'chain_length': 0,
                    'oracle_chain': [],
                    'frontier': [],
                    'source_meta': meta,
                })
            return 1
        for b in range(B):
            if self.repair_source_use_ar:
                cur_body = ar_init_bodies.get(int(b), gt_bodies[b])
                ar_state_count += append_training_state(
                    b,
                    cur_body,
                    source_type='ar',
                    source_meta={
                        'desired_depth_distribution': '',
                        'desired_depth_sample': None,
                        'realized_corruption_depth': 0,
                        'corruption_chain': [],
                    },
                )
                if self.repair_source_use_rollout and self.repair_source_rollout_prob > 0.0 and random.random() < self.repair_source_rollout_prob:
                    with torch.no_grad():
                        rolled = torch.tensor([helper.pack(cur_body, max_len=L)], device=trg.device, dtype=torch.long)
                        for _ in range(self.repair_source_rollout_steps):
                            step_out = self._repair_one_step_global(
                                enc_out=enc_out[b:b + 1],
                                init_tokens=rolled,
                                conf_threshold=float(self.repair_conf_threshold),
                                editor_beam_k=1,
                                tagger_topk=1,
                                n_vars=n_vars,
                                trace_print=False,
                            )
                            if not step_out['successors']:
                                break
                            rolled = step_out['successors'][0]['tokens']
                        roll_body = helper.extract_body(rolled[0].detach().cpu().tolist())
                    if roll_body and helper.validate_body(roll_body) and [int(x) for x in roll_body] != [int(x) for x in cur_body]:
                        rollout_state_count += append_training_state(
                            b,
                            [int(x) for x in roll_body],
                            source_type='rollout',
                            source_meta={
                                'desired_depth_distribution': '',
                                'desired_depth_sample': None,
                                'realized_corruption_depth': 0,
                                'rollout_steps': int(self.repair_source_rollout_steps),
                                'corruption_chain': [],
                            },
                        )

            if self.repair_source_use_synth:
                desired_depth = int(random.randint(self.repair_chain_depth_min, max(self.repair_chain_depth_min, self.repair_chain_depth_max)))
                synth_budget = int(max(2, int(self.repair_direct_rewrite_max_nodes)))
                best_chain = []
                attempts_used = 0
                attempt_chain_lengths: List[int] = []
                for attempt_idx in range(self.repair_chain_resample_attempts):
                    attempts_used = int(attempt_idx + 1)
                    cand_chain = helper.sample_root_corruption_chain(
                        gt_body=gt_bodies[b],
                        step_idx=desired_depth,
                        max_body_len=max_body_len,
                        T_max=max(desired_depth, 1),
                        rewrite_budget=synth_budget,
                    )
                    attempt_chain_lengths.append(int(len(cand_chain)))
                    if len(cand_chain) > len(best_chain):
                        best_chain = cand_chain
                    if len(cand_chain) >= desired_depth:
                        best_chain = cand_chain[:desired_depth]
                        break
                if best_chain:
                    source_body = [int(x) for x in best_chain[-1].cur_body[:max_body_len]]
                    synth_state_count += append_training_state(
                        b,
                        source_body,
                        source_type='synth',
                        source_meta={
                            'desired_depth_distribution': f"uniform_int[{int(self.repair_chain_depth_min)},{int(max(self.repair_chain_depth_min, self.repair_chain_depth_max))}]",
                            'desired_depth_sample': int(desired_depth),
                            'realized_corruption_depth': int(len(best_chain)),
                            'resample_attempts_max': int(self.repair_chain_resample_attempts),
                            'resample_attempts_used': int(attempts_used),
                            'attempt_chain_lengths': [int(x) for x in attempt_chain_lengths],
                            'corruption_chain': [
                                {
                                    'step_idx': int(rec.step_idx),
                                    'forward_op': str(rec.forward_op),
                                    'root_idx': int(rec.root_idx),
                                    'prev_span': [int(x) for x in rec.prev_span],
                                    'cur_span': [int(x) for x in rec.cur_span],
                                    'prev_body': [int(x) for x in rec.prev_body[:max_body_len]],
                                    'cur_body': [int(x) for x in rec.cur_body[:max_body_len]],
                                    'prev_subtree': [int(x) for x in rec.prev_subtree],
                                    'prev_root_token': int(rec.prev_root_token) if rec.prev_root_token is not None else None,
                                }
                                for rec in best_chain
                            ],
                        },
                    )

        if self.repair_tagger_keep_self_prob > 0.0:
            for b in range(B):
                if random.random() < float(self.repair_tagger_keep_self_prob):
                    keep_self_state_count += append_keep_state(
                        b,
                        gt_bodies[b],
                        source_type='gt_keep',
                        source_meta={
                            'keep_state_probability': float(self.repair_tagger_keep_self_prob),
                            'keep_state_target': 'gt_to_gt',
                        },
                    )

        if self._repair_trace_is_enabled() and trace_candidates and self._repair_trace_cases_written < self.repair_trace_max_cases:
            remaining = int(max(0, self.repair_trace_max_cases - self._repair_trace_cases_written))
            take = int(min(len(trace_candidates), self.repair_trace_cases_per_batch, remaining))
            if take > 0:
                chosen = random.sample(trace_candidates, k=take) if take < len(trace_candidates) else list(trace_candidates)
                rows_to_write: List[Dict[str, Any]] = []
                for payload in chosen:
                    rows_to_write.extend(self._repair_trace_rows_from_payload(payload))
                self._repair_trace_write_rows(rows_to_write)
                self._repair_trace_cases_written += int(take)

        states_total = int(len(state_cur_tokens))
        zero = torch.tensor(0.0, device=trg.device)
        zero_long = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        if states_total == 0:
            return {
                'repair_loss': zero,
                'tagger_loss': zero,
                'generator_loss': zero,
                'repair_states_total': torch.tensor(0.0, dtype=torch.float32, device=trg.device),
                'repair_ar_state_count': torch.tensor(float(ar_state_count), dtype=torch.float32, device=trg.device),
                'repair_synth_state_count': torch.tensor(float(synth_state_count), dtype=torch.float32, device=trg.device),
                'repair_rollout_state_count': torch.tensor(float(rollout_state_count), dtype=torch.float32, device=trg.device),
                'repair_keep_self_state_count': torch.tensor(float(keep_self_state_count), dtype=torch.float32, device=trg.device),
                'repair_tagger_overall_acc': zero,
                'repair_action_prop_keep': zero,
                'repair_action_prop_replace': zero,
                'repair_action_prop_delete': zero,
                'repair_action_prop_rewrite': zero,
                'repair_action_prop_insert': zero,
                'repair_action_acc_keep': zero,
                'repair_action_acc_replace': zero,
                'repair_action_acc_delete': zero,
                'repair_action_acc_rewrite': zero,
                'repair_action_acc_insert': zero,
                'repair_editor_loss_keep': zero,
                'repair_editor_loss_replace': zero,
                'repair_editor_loss_delete': zero,
                'repair_editor_loss_rewrite': zero,
                'repair_editor_loss_insert': zero,
                'repair_editor_acc_keep': zero,
                'repair_editor_acc_replace': zero,
                'repair_editor_acc_delete': zero,
                'repair_editor_acc_rewrite': zero,
                'repair_editor_acc_insert': zero,
                'repair_editor_token_count_keep': zero_long,
                'repair_editor_token_count_replace': zero_long,
                'repair_editor_token_count_delete': zero_long,
                'repair_editor_token_count_rewrite': zero_long,
                'repair_editor_token_count_insert': zero_long,
                'repair_editor_exact_overall': zero,
                'repair_editor_exact_replace': zero,
                'repair_editor_exact_delete': zero,
                'repair_editor_exact_rewrite': zero,
                'repair_editor_exact_insert': zero,
            }

        cur_tokens = torch.tensor(state_cur_tokens, dtype=torch.long, device=trg.device)
        action_tgt = torch.tensor(state_action_tgt, dtype=torch.long, device=trg.device)
        editor_orig_inp = torch.tensor(state_editor_orig_inp, dtype=torch.long, device=trg.device)
        editor_inp = torch.tensor(state_editor_inp, dtype=torch.long, device=trg.device)
        hole_mask = torch.tensor(state_hole_mask, dtype=torch.bool, device=trg.device)
        action_ids = torch.tensor(state_action_ids, dtype=torch.long, device=trg.device)
        gen_tgt = torch.tensor(state_gen_tgt, dtype=torch.long, device=trg.device)
        single_tgt = torch.tensor(state_single_tgt, dtype=torch.long, device=trg.device)
        root_pos_t = torch.tensor(state_root_pos, dtype=torch.long, device=trg.device)
        state_action_label_t = torch.tensor(state_action_label, dtype=torch.long, device=trg.device)
        enc_src_idx = torch.tensor(state_enc_src_idx, dtype=torch.long, device=trg.device)
        body_lens_t = torch.tensor(state_body_len, dtype=torch.long, device=trg.device).clamp(min=0, max=max_body_len)
        has_editor_t = torch.tensor(state_has_editor, dtype=torch.bool, device=trg.device)
        enc_out_states = enc_out.index_select(0, enc_src_idx)
        N = int(cur_tokens.shape[0])

        action_logits = self.repair_tagger_logits(tokens=cur_tokens, enc_out=enc_out_states)
        pos_idx = torch.arange(L, device=trg.device).unsqueeze(0)
        body_mask = ((pos_idx >= 1) & (pos_idx < (1 + body_lens_t.unsqueeze(1))))
        valid_total = body_mask.sum().clamp_min(1)
        tgt_valid = action_tgt[body_mask]
        class_counts = torch.bincount(tgt_valid, minlength=NUM_TAGGER_ACTIONS).float() if int(tgt_valid.numel()) > 0 else torch.zeros(NUM_TAGGER_ACTIONS, device=trg.device)
        class_props = class_counts / class_counts.sum().clamp_min(1.0)
        class_weights = torch.ones(NUM_TAGGER_ACTIONS, dtype=torch.float32, device=trg.device)
        if float(self.repair_tagger_class_balance_power) > 0.0:
            class_weights = class_props.clamp_min(float(self.repair_tagger_min_class_prop)).pow(-float(self.repair_tagger_class_balance_power))
            class_weights = class_weights / class_weights.mean().clamp_min(1e-6)
        class_weights = class_weights.clamp(max=float(self.repair_tagger_max_class_weight))
        class_weights[int(ACT_KEEP)] *= float(self.repair_tagger_keep_weight)
        class_weights[int(ACT_REPLACE)] *= float(self.repair_tagger_replace_weight)
        class_weights[int(ACT_DELETE_SUBTREE)] *= float(self.repair_tagger_delete_weight)
        class_weights[int(ACT_REWRITE_SUBTREE)] *= float(self.repair_tagger_rewrite_weight)
        class_weights[int(ACT_INSERT)] *= float(self.repair_tagger_insert_weight)

        tag_loss_raw = F.cross_entropy(
            action_logits.reshape(N * L, NUM_TAGGER_ACTIONS),
            action_tgt.reshape(N * L),
            reduction='none',
            weight=class_weights,
        ).view(N, L)
        tagger_loss = (tag_loss_raw * body_mask.float()).sum() / body_mask.float().sum().clamp_min(1.0)

        with torch.no_grad():
            pred_actions = action_logits.argmax(dim=-1)
            valid_mask = body_mask.bool()
            valid_total_f = valid_mask.sum().clamp_min(1).float()
            correct_mask = (pred_actions == action_tgt) & valid_mask
            repair_tagger_overall_acc = correct_mask.sum().float() / valid_total_f
            action_prop_keep = torch.tensor(0.0, device=trg.device)
            action_prop_replace = torch.tensor(0.0, device=trg.device)
            action_prop_delete = torch.tensor(0.0, device=trg.device)
            action_prop_rewrite = torch.tensor(0.0, device=trg.device)
            action_prop_insert = torch.tensor(0.0, device=trg.device)
            action_acc_keep = torch.tensor(0.0, device=trg.device)
            action_acc_replace = torch.tensor(0.0, device=trg.device)
            action_acc_delete = torch.tensor(0.0, device=trg.device)
            action_acc_rewrite = torch.tensor(0.0, device=trg.device)
            action_acc_insert = torch.tensor(0.0, device=trg.device)
            for act_id in TAGGER_ACTIONS:
                tgt_mask = (action_tgt == int(act_id)) & valid_mask
                tgt_count = tgt_mask.sum()
                prop_val = tgt_count.float() / valid_total_f
                acc_val = torch.tensor(0.0, device=trg.device)
                if int(tgt_count.item()) > 0:
                    acc_val = ((pred_actions == int(act_id)) & tgt_mask).sum().float() / tgt_count.float()
                if int(act_id) == int(ACT_KEEP):
                    action_prop_keep = prop_val
                    action_acc_keep = acc_val
                elif int(act_id) == int(ACT_REPLACE):
                    action_prop_replace = prop_val
                    action_acc_replace = acc_val
                elif int(act_id) == int(ACT_DELETE_SUBTREE):
                    action_prop_delete = prop_val
                    action_acc_delete = acc_val
                elif int(act_id) == int(ACT_REWRITE_SUBTREE):
                    action_prop_rewrite = prop_val
                    action_acc_rewrite = acc_val
                elif int(act_id) == int(ACT_INSERT):
                    action_prop_insert = prop_val
                    action_acc_insert = acc_val

        generator_loss = torch.tensor(0.0, device=trg.device)
        repair_editor_loss_keep = torch.tensor(0.0, device=trg.device)
        repair_editor_loss_replace = torch.tensor(0.0, device=trg.device)
        repair_editor_loss_delete = torch.tensor(0.0, device=trg.device)
        repair_editor_loss_rewrite = torch.tensor(0.0, device=trg.device)
        repair_editor_loss_insert = torch.tensor(0.0, device=trg.device)
        repair_editor_acc_keep = torch.tensor(0.0, device=trg.device)
        repair_editor_acc_replace = torch.tensor(0.0, device=trg.device)
        repair_editor_acc_delete = torch.tensor(0.0, device=trg.device)
        repair_editor_acc_rewrite = torch.tensor(0.0, device=trg.device)
        repair_editor_acc_insert = torch.tensor(0.0, device=trg.device)
        repair_editor_token_count_keep = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        repair_editor_token_count_replace = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        repair_editor_token_count_delete = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        repair_editor_token_count_rewrite = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        repair_editor_token_count_insert = torch.tensor(0.0, dtype=torch.float32, device=trg.device)
        repair_editor_exact_overall = torch.tensor(0.0, device=trg.device)
        repair_editor_exact_replace = torch.tensor(0.0, device=trg.device)
        repair_editor_exact_delete = torch.tensor(0.0, device=trg.device)
        repair_editor_exact_rewrite = torch.tensor(0.0, device=trg.device)
        repair_editor_exact_insert = torch.tensor(0.0, device=trg.device)
        if bool(has_editor_t.any().item()):
            per_state_loss = torch.zeros((N,), dtype=torch.float32, device=trg.device)
            per_state_exact_any = torch.zeros((N,), dtype=torch.bool, device=trg.device)

            gen_state_mask = has_editor_t & ((state_action_label_t == int(ACT_INSERT)) | (state_action_label_t == int(ACT_REWRITE_SUBTREE)))
            if bool(gen_state_mask.any().item()):
                editor_logits = self.repair_generator_logits(
                    orig_tokens=editor_orig_inp,
                    edit_tokens=editor_inp,
                    enc_out=enc_out_states,
                    action_ids=action_ids,
                    hole_mask=hole_mask,
                )
                gen_mask = (gen_tgt != -100) & gen_state_mask.unsqueeze(1)
                safe_tgt = gen_tgt.clone()
                safe_tgt[~gen_mask] = 0
                loss_flat = F.cross_entropy(
                    editor_logits.reshape(N * L, editor_logits.shape[-1]),
                    safe_tgt.reshape(N * L),
                    reduction='none',
                ).view(N, L)
                per_state_gen = (loss_flat * gen_mask.float()).sum(dim=1) / gen_mask.float().sum(dim=1).clamp_min(1.0)
                per_state_loss = torch.where(gen_state_mask, per_state_gen, per_state_loss)

                with torch.no_grad():
                    pred_tok = editor_logits.argmax(dim=-1)
                    same_tok = ((pred_tok == safe_tgt) | (~gen_mask))
                    per_state_exact_gen = same_tok.all(dim=1) & gen_state_mask
                    per_state_exact_any = per_state_exact_any | per_state_exact_gen
                    repair_editor_exact_overall = per_state_exact_any[has_editor_t].float().mean()
                    for act in (ACT_REWRITE_SUBTREE, ACT_INSERT):
                        act_mask = gen_mask & (action_ids == int(act))
                        act_count = act_mask.sum().float()
                        act_loss = torch.tensor(0.0, device=trg.device)
                        act_acc = torch.tensor(0.0, device=trg.device)
                        act_exact = torch.tensor(0.0, device=trg.device)
                        state_mask = gen_state_mask & (state_action_label_t == int(act))
                        if float(act_count.item()) > 0:
                            act_loss = (loss_flat * act_mask.float()).sum() / act_count.clamp_min(1.0)
                            act_acc = (((pred_tok == safe_tgt) & act_mask).sum().float() / act_count.clamp_min(1.0))
                        if bool(state_mask.any().item()):
                            act_exact = per_state_exact_gen[state_mask].float().mean()
                        if int(act) == int(ACT_REWRITE_SUBTREE):
                            repair_editor_loss_rewrite = act_loss
                            repair_editor_acc_rewrite = act_acc
                            repair_editor_token_count_rewrite = act_count
                            repair_editor_exact_rewrite = act_exact
                        else:
                            repair_editor_loss_insert = act_loss
                            repair_editor_acc_insert = act_acc
                            repair_editor_token_count_insert = act_count
                            repair_editor_exact_insert = act_exact

            for act in (ACT_REPLACE, ACT_DELETE_SUBTREE):
                cls_mask = has_editor_t & (state_action_label_t == int(act)) & (single_tgt != -100)
                if not bool(cls_mask.any().item()):
                    continue
                logits = self.repair_replace_delete_logits(
                    orig_tokens=editor_orig_inp[cls_mask],
                    edit_tokens=editor_inp[cls_mask],
                    enc_out=enc_out_states[cls_mask],
                    action_ids=action_ids[cls_mask],
                    hole_mask=hole_mask[cls_mask],
                    root_positions=root_pos_t[cls_mask],
                    action_label=int(act),
                )
                ce = F.cross_entropy(logits, single_tgt[cls_mask], reduction='none')
                per_state_loss[cls_mask] = ce
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    acc_vec = (pred == single_tgt[cls_mask])
                    per_state_exact_any[cls_mask] = acc_vec
                    if bool(has_editor_t.any().item()):
                        repair_editor_exact_overall = per_state_exact_any[has_editor_t].float().mean()
                    if int(act) == int(ACT_REPLACE):
                        repair_editor_loss_replace = ce.mean()
                        repair_editor_acc_replace = acc_vec.float().mean()
                        repair_editor_token_count_replace = cls_mask.sum().float()
                        repair_editor_exact_replace = acc_vec.float().mean()
                    else:
                        repair_editor_loss_delete = ce.mean()
                        repair_editor_acc_delete = acc_vec.float().mean()
                        repair_editor_token_count_delete = cls_mask.sum().float()
                        repair_editor_exact_delete = acc_vec.float().mean()

            generator_loss = per_state_loss[has_editor_t].mean() if bool(has_editor_t.any().item()) else torch.tensor(0.0, device=trg.device)
        repair_loss = tagger_loss + generator_loss
        return {
            'repair_loss': repair_loss,
            'tagger_loss': tagger_loss,
            'generator_loss': generator_loss,
            'repair_states_total': torch.tensor(float(states_total), dtype=torch.float32, device=trg.device),
            'repair_ar_state_count': torch.tensor(float(ar_state_count), dtype=torch.float32, device=trg.device),
            'repair_synth_state_count': torch.tensor(float(synth_state_count), dtype=torch.float32, device=trg.device),
            'repair_rollout_state_count': torch.tensor(float(rollout_state_count), dtype=torch.float32, device=trg.device),
            'repair_keep_self_state_count': torch.tensor(float(keep_self_state_count), dtype=torch.float32, device=trg.device),
            'repair_tagger_overall_acc': repair_tagger_overall_acc,
            'repair_action_prop_keep': action_prop_keep,
            'repair_action_prop_replace': action_prop_replace,
            'repair_action_prop_delete': action_prop_delete,
            'repair_action_prop_rewrite': action_prop_rewrite,
            'repair_action_prop_insert': action_prop_insert,
            'repair_action_acc_keep': action_acc_keep,
            'repair_action_acc_replace': action_acc_replace,
            'repair_action_acc_delete': action_acc_delete,
            'repair_action_acc_rewrite': action_acc_rewrite,
            'repair_action_acc_insert': action_acc_insert,
            'repair_editor_loss_keep': repair_editor_loss_keep,
            'repair_editor_loss_replace': repair_editor_loss_replace,
            'repair_editor_loss_delete': repair_editor_loss_delete,
            'repair_editor_loss_rewrite': repair_editor_loss_rewrite,
            'repair_editor_loss_insert': repair_editor_loss_insert,
            'repair_editor_acc_keep': repair_editor_acc_keep,
            'repair_editor_acc_replace': repair_editor_acc_replace,
            'repair_editor_acc_delete': repair_editor_acc_delete,
            'repair_editor_acc_rewrite': repair_editor_acc_rewrite,
            'repair_editor_acc_insert': repair_editor_acc_insert,
            'repair_editor_token_count_keep': repair_editor_token_count_keep,
            'repair_editor_token_count_replace': repair_editor_token_count_replace,
            'repair_editor_token_count_delete': repair_editor_token_count_delete,
            'repair_editor_token_count_rewrite': repair_editor_token_count_rewrite,
            'repair_editor_token_count_insert': repair_editor_token_count_insert,
            'repair_editor_exact_overall': repair_editor_exact_overall,
            'repair_editor_exact_replace': repair_editor_exact_replace,
            'repair_editor_exact_delete': repair_editor_exact_delete,
            'repair_editor_exact_rewrite': repair_editor_exact_rewrite,
            'repair_editor_exact_insert': repair_editor_exact_insert,
        }

    def repair_candidate_pool_2d_beam(
            self,
            *,
            enc_out: torch.Tensor,
            init_tokens: torch.Tensor,
            use_repair: bool = True,
            steps: Optional[int] = None,
            conf_threshold: Optional[float] = None,
            editor_beam_k: int = 4,
            tagger_topk: int = 1,
            revision_beam_k: int = 4,
            n_vars: Optional[int] = None,
            trace_print: bool = False,
    ) -> List[Dict[str, Any]]:
        self._maybe_init_vocab_helper()
        helper = self._repair_helper
        assert helper is not None
        if (not use_repair) or init_tokens is None:
            tok = self._truncate_after_F(init_tokens[:1].long().to(enc_out.device))
            return [{'tokens': tok, 'score': 0.0, 'raw_logp': 0.0, 'done': True, 'trace': []}]
        if init_tokens.dim() != 2 or init_tokens.shape[0] != 1:
            raise ValueError('repair_candidate_pool_2d_beam expects batch size 1')
        steps = int(steps) if steps is not None else 8
        conf_threshold = float(conf_threshold) if conf_threshold is not None else float(self.repair_conf_threshold)
        editor_beam_k = int(max(1, editor_beam_k))
        tagger_topk = int(max(1, tagger_topk))
        revision_beam_k = int(max(1, revision_beam_k))
        n_vars_eff = int(n_vars) if n_vars is not None else int(getattr(self.cfg, 'num_features', 0))

        tok0 = self._truncate_after_F(init_tokens[:1].long().to(enc_out.device))
        init_key = tuple(int(x) for x in tok0[0].detach().cpu().tolist())
        beam = [{'tokens': tok0, 'score': 0.0, 'raw_logp': 0.0, 'done': False, 'trace': []}]
        pool: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        def _fmt_tokens(tok_row: torch.Tensor) -> str:
            body = helper.extract_body(tok_row.detach().cpu().tolist())
            return ' '.join([str(helper.id2word.get(int(t), int(t))) for t in body])

        for s in range(int(steps)):
            next_all: List[Dict[str, Any]] = []
            for cand in beam:
                ctok = cand['tokens']
                ctr = list(cand.get('trace', []))
                if bool(cand.get('done', False)):
                    next_all.append(cand)
                    continue
                step_out = self._repair_one_step_global(
                    enc_out=enc_out[:1],
                    init_tokens=ctok,
                    conf_threshold=float(conf_threshold),
                    editor_beam_k=int(editor_beam_k),
                    tagger_topk=int(tagger_topk),
                    n_vars=n_vars_eff,
                    trace_print=False,
                )
                if not step_out['successors']:
                    step_info = {
                        'step': int(s),
                        'input_expr': _fmt_tokens(ctok[0]),
                        'best_nonkeep': float(step_out.get('best_nonkeep', 0.0)),
                        'reason': str(step_out.get('reason', 'no_successor')),
                    }
                    next_all.append({
                        'tokens': ctok,
                        'score': float(cand.get('score', 0.0)),
                        'raw_logp': float(cand.get('raw_logp', 0.0)),
                        'done': True,
                        'trace': ctr + [step_info],
                    })
                    continue
                for succ in step_out['successors'][:editor_beam_k]:
                    step_info = {
                        'step': int(s),
                        'input_expr': _fmt_tokens(ctok[0]),
                        'selected_edit': (int(succ['selected_edit']['root_idx']), int(succ['selected_edit']['action'])) if succ.get('selected_edit', None) is not None else None,
                        'best_nonkeep': float(succ.get('best_nonkeep', 0.0)),
                        'tag_logp': float(succ.get('tag_logp', 0.0)),
                        'editor_logp': float(succ.get('editor_logp', 0.0)),
                        'output_expr': _fmt_tokens(succ['tokens'][0]),
                    }
                    raw2 = float(cand.get('raw_logp', 0.0) + succ['raw_logp'])
                    next_all.append({
                        'tokens': succ['tokens'],
                        'score': raw2,
                        'raw_logp': raw2,
                        'done': False,
                        'trace': ctr + [step_info],
                    })
            if not next_all:
                break
            dedup: Dict[Tuple[int, ...], Dict[str, Any]] = {}
            for cand in next_all:
                key = tuple(int(x) for x in cand['tokens'][0].detach().cpu().tolist())
                prev = dedup.get(key)
                if (prev is None) or (float(cand.get('score', -1e18)) > float(prev.get('score', -1e18))):
                    dedup[key] = cand
                if key != init_key:
                    pool_prev = pool.get(key)
                    if (pool_prev is None) or (float(cand.get('score', -1e18)) > float(pool_prev.get('score', -1e18))):
                        pool[key] = cand
            beam = sorted(dedup.values(), key=lambda d: float(d.get('score', -1e18)), reverse=True)[:revision_beam_k]
            if all(bool(c.get('done', False)) for c in beam):
                break

        if not pool:
            return []
        final_pool = sorted(pool.values(), key=lambda d: float(d.get('score', -1e18)), reverse=True)
        if trace_print and len(final_pool) > 0:
            best = final_pool[0]
            print('[repair_trace_2d] init =', _fmt_tokens(tok0[0]))
            for item in list(best.get('trace', [])):
                if item.get('reason', None) is not None:
                    print('[repair_trace_2d] step=%d stop=%s best_nonkeep=%s input=%s' % (
                        int(item.get('step', -1)), str(item.get('reason')), str(item.get('best_nonkeep', None)), str(item.get('input_expr', ''))))
                else:
                    print('[repair_trace_2d] step=%d selected=%s tag_lp=%.4f editor_lp=%.4f' % (
                        int(item.get('step', -1)), str(item.get('selected_edit', None)), float(item.get('tag_logp', 0.0)), float(item.get('editor_logp', 0.0))))
                    print('[repair_trace_2d]   input =', str(item.get('input_expr', '')))
                    print('[repair_trace_2d]   output=', str(item.get('output_expr', '')))
            print('[repair_trace_2d] final =', _fmt_tokens(best['tokens'][0]))
        return final_pool

    def repair_refine_tokens_edit_beam(
            self,
            enc_out: torch.Tensor,
            init_tokens: torch.Tensor,
            *,
            use_repair: bool = True,
            steps: Optional[int] = None,
            conf_threshold: Optional[float] = None,
            editor_beam_k: int = 4,
            tagger_topk: int = 1,
            revision_beam_k: int = 4,
            n_vars: Optional[int] = None,
            trace_print: bool = False,
    ) -> torch.Tensor:
        tokens = init_tokens if init_tokens.dim() == 2 else init_tokens.unsqueeze(0)
        enc = enc_out if enc_out.dim() == 3 else enc_out.unsqueeze(0)
        outs: List[torch.Tensor] = []
        for b in range(tokens.shape[0]):
            pool = self.repair_candidate_pool_2d_beam(
                enc_out=enc[b:b + 1],
                init_tokens=tokens[b:b + 1],
                use_repair=use_repair,
                steps=steps,
                conf_threshold=conf_threshold,
                editor_beam_k=editor_beam_k,
                tagger_topk=tagger_topk,
                revision_beam_k=revision_beam_k,
                n_vars=n_vars,
                trace_print=trace_print,
            )
            if pool:
                outs.append(pool[0]['tokens'])
            else:
                outs.append(self._truncate_after_F(tokens[b:b + 1].long().to(enc.device)))
        return self._truncate_after_F(torch.cat(outs, dim=0))

    def fitfunc2(self, X, y, cfg_params=None, test_data=None):
        if cfg_params is None:
            raise ValueError("fitfunc2 requires cfg_params (pass cfg.inference).")
        print("beam = ", cfg_params.beam_size)

        # --------------------------------------------------
        # Determinism hooks for evaluation / ablations
        #   - eval_seed can be injected from test scripts.
        #   - We set both torch and numpy/random so that:
        #       (1) point subsampling (randperm) is identical across runs
        #       (2) BFGS restarts are reproducible (via bfgs_wrapper seeds)
        # --------------------------------------------------
        eval_seed = int(cfg_params.eval_seed)
        if eval_seed != 0:
            try:
                torch.manual_seed(eval_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(eval_seed)
                np.random.seed(eval_seed % (2 ** 32 - 1))
                random.seed(eval_seed % (2 ** 32 - 1))
            except Exception:
                pass

        # ==========================================
        # 1. 动态获取 ID & 定义集合
        # ==========================================
        w2i = test_data.word2id

        # Cache vocab inside model for repair/corruption (from data.py)
        if (not getattr(self, '_vocab_ready', False)) and hasattr(test_data, 'word2id') and hasattr(test_data,
                                                                                                    'id2word'):
            self.word2id = dict(test_data.word2id)
            self.id2word = dict(test_data.id2word)
            self._maybe_init_vocab_helper()

        use_repair = bool(cfg_params.use_repair)
        return_baseline_and_repair = bool(cfg_params.return_baseline_and_repair)

        repair_iters = int(cfg_params.repair_max_iters)
        repair_conf = float(cfg_params.repair_conf)

        # Use the vocab-driven grammar helper to avoid operator-set drift (fitfunc2 vs PrefixRepairHelper).
        helper = getattr(self, "_repair_helper", None)
        if helper is None and isinstance(w2i, dict) and hasattr(test_data, 'id2word'):
            helper = PrefixRepairHelper(w2i, test_data.id2word)
        if helper is not None:
            arity_1_ids = {int(x) for x in getattr(helper, "unary_ids", [])}
            arity_2_ids = {int(x) for x in getattr(helper, "binary_ids", [])}
        else:
            arity_1_ids = set()
            arity_2_ids = set()
        all_op_ids = arity_1_ids | arity_2_ids

        # Keep previous behavior: do NOT enforce transcendental nesting constraints here unless explicitly configured.
        transcendental_ids = set()
        pow_id = w2i.get('pow')

        # 开关逻辑
        limit_pow_const = bool(cfg_params.no_c_in_pow)
        if limit_pow_const:
            c_id = w2i.get('c', 3)
            print("Info: Constraint [No Constant in Pow] is ENABLED.")
        else:
            c_id = None

        pad_id = w2i.get('P', 0)
        start_id = w2i.get('S', 1)
        finish_id = w2i.get('F', 2)

        # ==========================================
        # 2. 数据准备 & 变量屏蔽
        # ==========================================
        X = X
        # n_vars must be derived from the *unbatched* X (shape: [n_points, n_vars])
        n_vars = int(X.shape[1])
        y = y[:, None]
        X = X.clone().detach().to(cfg_params.device).unsqueeze(0)
        if X.shape[2] < self.cfg.dim_input - 1:
            pad = torch.zeros(1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=cfg_params.device)
            X = torch.cat((X, pad), dim=2)

        input_X = X[0, :, :10]
        abs_sum = torch.abs(input_X).sum(dim=0)
        unused_feat_indices = (abs_sum == 0).nonzero(as_tuple=True)[0].cpu().numpy()
        masked_var_ids = set()
        for idx in unused_feat_indices:
            var_name = f"x_{idx + 1}"
            if var_name in w2i:
                masked_var_ids.add(w2i[var_name])
            elif f"x_{idx}" in w2i:
                masked_var_ids.add(w2i[f"x_{idx}"])

        y = y.clone().detach().to(cfg_params.device).unsqueeze(0)

        # ==========================================
        # 3. Beam Search 主循环
        # ==========================================
        with torch.no_grad():
            # 随机采样
            n_points = X.shape[1]
            if n_points > 200:
                indices = torch.randperm(n_points, device=cfg_params.device)[:200]
                indices, _ = torch.sort(indices)
                X_enc = X[:, indices, :]
                y_enc = y[:, indices, :]
                encoder_input = torch.cat((X_enc, y_enc), dim=2)
            else:
                encoder_input = torch.cat((X, y), dim=2)

            src_enc = self.MultiModalEncoder(encoder_input)

            shape_enc_src = (cfg_params.beam_size,) + src_enc.shape[1:]
            enc_src = src_enc.unsqueeze(1).expand((1, cfg_params.beam_size) + src_enc.shape[1:]).contiguous().view(
                shape_enc_src)

            generated = torch.zeros([cfg_params.beam_size, self.cfg.length_eq], dtype=torch.long,
                                    device=cfg_params.device)
            generated[:, 0] = start_id

            generated_hyps = BeamHypotheses(cfg_params.beam_size, self.cfg.length_eq, 1.0, 1)
            done = False
            beam_scores = torch.zeros(cfg_params.beam_size, device=cfg_params.device, dtype=torch.float)
            beam_scores[1:] = -1e9

            cur_len = torch.tensor(1, device=cfg_params.device, dtype=torch.int64)
            cache = {"slen": 0}

            while cur_len < self.cfg.length_eq:
                generated_kpm, generated_causal = self.make_trg_mask(generated[:, :cur_len])
                pos = self.pos_embedding(
                    torch.arange(0, cur_len, device=generated.device).unsqueeze(0).repeat(generated.shape[0],
                                                                                          1).type_as(generated))
                te = self.tok_embedding(generated[:, :cur_len])
                trg_ = self.dropout(te + pos)
                output = self.decoder_transfomer(
                    trg_.permute(1, 0, 2), enc_src.permute(1, 0, 2),
                    generated_causal, tgt_key_padding_mask=generated_kpm
                )
                output = self.fc_out(output).permute(1, 0, 2).contiguous()
                scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
                scores = self._mask_forbidden_output_logits(scores)
                n_words = scores.shape[-1]

                # --- Constraint Logic ---
                logit_mask = torch.zeros_like(scores)

                for i in range(cfg_params.beam_size):
                    if beam_scores[i] < -1e8:
                        continue

                    hyp_seq = generated[i, :cur_len].cpu().tolist()

                    valency, forbidden_by_structure = self._analyze_prefix_tree_context(
                        hyp_seq, arity_1_ids, arity_2_ids,
                        transcendental_ids, pow_id, c_id, start_id
                    )

                    # If the prefix already forms a complete tree (valency==0), we should stop
                    # generating body tokens and force the finish token. This eliminates the
                    # "complete-then-drift" failure mode and guarantees syntactically closed outputs.
                    if valency == 0 and finish_id < n_words:
                        logit_mask[i, :] = float('-inf')
                        logit_mask[i, finish_id] = 0.0
                        continue

                    forbidden = set()
                    forbidden.update(forbidden_by_structure)

                    remaining_len = self.cfg.length_eq - cur_len.item()
                    if valency >= remaining_len:
                        forbidden.update(all_op_ids)

                    if valency > 0:
                        forbidden.add(finish_id)
                        forbidden.add(pad_id)

                    forbidden.update(masked_var_ids)

                    if forbidden:
                        valid_forbidden = [x for x in forbidden if x < n_words]
                        if valid_forbidden:
                            logit_mask[i, valid_forbidden] = float('-inf')

                scores = scores + logit_mask
                # ------------------------

                _scores = scores + beam_scores[:, None].expand_as(scores)
                _scores = _scores.view(cfg_params.beam_size * n_words)
                next_scores, next_words = torch.topk(_scores, 2 * cfg_params.beam_size, dim=0, largest=True,
                                                     sorted=True)

                done = done or generated_hyps.is_done(next_scores.max().item())
                next_sent_beam = []
                for idx, value in zip(next_words, next_scores):
                    beam_id = idx // n_words
                    word_id = idx % n_words

                    # [修复] 只有当 word_id 是 Finish 时才加入结果集。

                    if word_id == finish_id:
                        generated_hyps.add(generated[beam_id, :cur_len].clone().cpu(), value.item())
                    else:
                        next_sent_beam.append((value, word_id, beam_id))
                    if len(next_sent_beam) == cfg_params.beam_size:
                        break

                if len(next_sent_beam) == 0:
                    # If every candidate ended with FINISH, we can stop decoding early.
                    if hasattr(generated_hyps, 'hyp') and len(generated_hyps.hyp) > 0:
                        break
                    next_sent_beam = [(0, self.trg_pad_idx, 0)] * cfg_params.beam_size

                beam_scores = torch.tensor([x[0] for x in next_sent_beam], device=cfg_params.device)
                beam_words = torch.tensor([x[1] for x in next_sent_beam], device=cfg_params.device)
                beam_idx = torch.tensor([x[2] for x in next_sent_beam], device=cfg_params.device)
                generated = generated[beam_idx, :]
                generated[:, cur_len] = beam_words
                for k in cache.keys():
                    if k != "slen":
                        cache[k] = (cache[k][0][beam_idx], cache[k][1][beam_idx])
                cur_len = cur_len + 1

            # ----------------------------
            # BFGS on (baseline beams) and optionally on (repaired beams)
            #   - return_baseline_and_repair=True: run both and return both
            #   - else: keep original behavior (if use_repair, run BFGS on repaired)
            # ----------------------------

            if 3 in test_data.id2word:
                test_data.id2word[3] = "constant"
            elif 'c' in w2i:
                test_data.id2word[w2i['c']] = "constant"

            X_cpu = X.cpu()
            y_cpu = y.cpu()
            sorted_hyps = sorted(generated_hyps.hyp, key=lambda x: x[0], reverse=True)

            # =========================================================
            # Strict validity filtering (valency must close)
            # =========================================================
            valid_hyps = []
            for score, ww in sorted_hyps:
                if isinstance(ww, torch.Tensor):
                    seq = ww.cpu().tolist()
                elif isinstance(ww, np.ndarray):
                    seq = ww.tolist()
                else:
                    seq = list(ww)

                if finish_id in seq:
                    seq = seq[:seq.index(finish_id)]
                seq = [s for s in seq if s != pad_id]

                valency, _ = self._analyze_prefix_tree_context(
                    seq, arity_1_ids, arity_2_ids,
                    transcendental_ids, pow_id, c_id, start_id
                )
                if valency == 0:
                    valid_hyps.append((score, ww))

            # Keep a snapshot for clean ablations

            # Keep a snapshot for clean ablations
            valid_hyps_baseline = list(valid_hyps)

            # Repair is now BFGS-guided below: after each edit step we evaluate candidates on the
            # current training set and use that value signal both for MSE-based early stopping and for beam ranking.

            bfgs_enabled = bool(getattr(cfg_params.bfgs, "activated", True))
            bfgs_max_workers = int(max(1, int(getattr(cfg_params.bfgs, "max_workers", 20))))
            repair_stop_mse = float(getattr(cfg_params, 'repair_bfgs_stop_mse', 1.0e-4))
            repair_skip_baseline_r2 = float(getattr(cfg_params, 'repair_skip_if_baseline_r2_ge', 1.0))

            def _token_to_key(tok_like: Any) -> Tuple[int, ...]:
                if tok_like is None:
                    return tuple()
                if isinstance(tok_like, torch.Tensor):
                    seq = tok_like.detach().cpu().tolist()
                elif isinstance(tok_like, np.ndarray):
                    seq = tok_like.tolist()
                else:
                    seq = list(tok_like)
                return tuple(int(x) for x in seq)

            def _normalize_seq(seq_like: Any) -> List[int]:
                if isinstance(seq_like, torch.Tensor):
                    seq = seq_like.detach().cpu().tolist()
                elif isinstance(seq_like, np.ndarray):
                    seq = seq_like.tolist()
                else:
                    seq = list(seq_like)
                seq = [int(s) for s in seq if int(s) != int(pad_id)]
                if finish_id in seq:
                    seq = seq[:seq.index(finish_id)]
                if len(seq) == 0 or seq[0] != start_id:
                    seq = [start_id] + seq
                return seq

            def _pack_init_from_seq(seq_like: Any) -> torch.Tensor:
                seq = _normalize_seq(seq_like)
                init = torch.full((1, self.cfg.length_eq), int(pad_id), dtype=torch.long, device=cfg_params.device)
                packed = (seq + [finish_id])[: self.cfg.length_eq]
                init[0, :len(packed)] = torch.tensor(packed, dtype=torch.long, device=cfg_params.device)
                return init

            def _compute_train_r2(expr_str: Optional[str]) -> float:
                if expr_str is None:
                    return float('nan')
                try:
                    expr = sp.sympify(str(expr_str))
                    y_true = np.asarray(y_cpu.detach().cpu().numpy()).reshape(-1)
                    values = {x: X_cpu[:, :, idx].detach().cpu().numpy() for idx, x in enumerate(test_data.total_variables)}
                    y_found = sp.lambdify(",".join(test_data.total_variables), expr, modules=bfgs.modules)(**values)
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

            def _pick_best_index(r2_list: List[float], loss_list: List[float]) -> Optional[int]:
                best_idx = None
                best_key = None
                for i, (r2_val, loss_val) in enumerate(zip(r2_list, loss_list)):
                    r2_finite = np.isfinite(r2_val)
                    loss_finite = np.isfinite(loss_val)
                    key = (
                        1 if r2_finite else 0,
                        float(r2_val) if r2_finite else -1.0e18,
                        1 if loss_finite else 0,
                        -float(loss_val) if loss_finite else -1.0e18,
                    )
                    if best_idx is None or key > best_key:
                        best_idx = int(i)
                        best_key = key
                return best_idx

            def _empty_eval_result(token_fallback=None):
                return {
                    'all_bfgs_preds': [None],
                    'all_bfgs_loss': [float('nan')],
                    'all_bfgs_r2': [float('nan')],
                    'best_bfgs_preds': [None],
                    'best_bfgs_loss': [float('nan')],
                    'best_bfgs_r2': [float('nan')],
                    'best_token': [token_fallback],
                    'all_tokens': [token_fallback],
                }

            def _summarize_eval_records(records: List[Dict[str, Any]]):
                if len(records) == 0:
                    return _empty_eval_result(token_fallback=None)
                preds = [rec.get('pred_expr', None) for rec in records]
                losses = [float(rec.get('bfgs_loss', float('nan'))) for rec in records]
                r2s = [float(rec.get('bfgs_r2', float('nan'))) for rec in records]
                tokens = [rec.get('token', None) for rec in records]
                best_idx = _pick_best_index(r2s, losses)
                if best_idx is None:
                    return _empty_eval_result(token_fallback=(tokens[0] if len(tokens) > 0 else None))
                return {
                    'all_bfgs_preds': preds,
                    'all_bfgs_loss': losses,
                    'all_bfgs_r2': r2s,
                    'best_bfgs_preds': [preds[best_idx]],
                    'best_bfgs_loss': [losses[best_idx]],
                    'best_bfgs_r2': [r2s[best_idx]],
                    'best_token': [tokens[best_idx]],
                    'all_tokens': tokens,
                }

            def _run_bfgs_on_hyps(valid_hyps_in, seed_offset: int = 0):
                if not bfgs_enabled:
                    token_list = [ww for _, ww in valid_hyps_in] if len(valid_hyps_in) > 0 else [None]
                    return _empty_eval_result(token_fallback=(token_list[0] if len(token_list) > 0 else None))

                task_list = [
                    (ww, X_cpu, y_cpu, cfg_params, test_data, int(eval_seed + seed_offset + i))
                    for i, (_, ww) in enumerate(valid_hyps_in)
                ]
                if len(task_list) == 0 and len(sorted_hyps) > 0:
                    task_list = [(sorted_hyps[0][1], X_cpu, y_cpu, cfg_params, test_data, int(eval_seed + seed_offset))]
                if len(task_list) == 0:
                    return _empty_eval_result(token_fallback=None)

                records: List[Dict[str, Any]] = []
                with ProcessPoolExecutor(bfgs_max_workers) as executor:
                    futures = [executor.submit(bfgs_wrapper, args) for args in task_list]
                    for future in as_completed(futures):
                        pred_w_c, loss_bfgs, ww_ = future.result()
                        if pred_w_c is None:
                            continue
                        records.append({
                            'pred_expr': pred_w_c,
                            'bfgs_loss': float(loss_bfgs),
                            'bfgs_r2': float(_compute_train_r2(pred_w_c)),
                            'token': ww_,
                        })
                return _summarize_eval_records(records)

            def _better_record(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
                if b is None:
                    return False
                if a is None:
                    return True
                ar2 = float(a.get('bfgs_r2', float('nan')))
                br2 = float(b.get('bfgs_r2', float('nan')))
                aloss = float(a.get('bfgs_loss', float('nan')))
                bloss = float(b.get('bfgs_loss', float('nan')))
                akey = (1 if np.isfinite(ar2) else 0, ar2 if np.isfinite(ar2) else -1.0e18,
                        1 if np.isfinite(aloss) else 0, -aloss if np.isfinite(aloss) else -1.0e18,
                        float(a.get('score', -1.0e18)))
                bkey = (1 if np.isfinite(br2) else 0, br2 if np.isfinite(br2) else -1.0e18,
                        1 if np.isfinite(bloss) else 0, -bloss if np.isfinite(bloss) else -1.0e18,
                        float(b.get('score', -1.0e18)))
                return bkey > akey

            def _guided_repair_search_greedy_chains(seed_hyps: List[Tuple[float, Any]]):
                active_states: List[Dict[str, Any]] = []
                for chain_rank, (seed_score, seed_tokens) in enumerate(seed_hyps):
                    init = _pack_init_from_seq(seed_tokens)
                    active_states.append({
                        'tokens': init,
                        'score': float(seed_score),
                        'raw_logp': 0.0,
                        'trace': [],
                        'chain_rank': int(chain_rank),
                        'last_key': _token_to_key(init[0]),
                    })

                all_records: Dict[Tuple[int, ...], Dict[str, Any]] = {}
                best_record: Optional[Dict[str, Any]] = None
                executed_iters = 0

                for step_idx in range(int(repair_iters)):
                    step_candidates: List[Dict[str, Any]] = []
                    next_active_states: List[Dict[str, Any]] = []

                    for state in active_states:
                        step_out = self._repair_one_step_global(
                            enc_out=src_enc,
                            init_tokens=state['tokens'],
                            conf_threshold=repair_conf,
                            editor_beam_k=1,
                            tagger_topk=1,
                            n_vars=n_vars,
                            trace_print=False,
                        )
                        if not step_out['successors']:
                            continue

                        succ = step_out['successors'][0]
                        tok = succ['tokens'][0].detach().cpu()
                        key = _token_to_key(tok)
                        if (not key) or key == state.get('last_key'):
                            continue

                        step_candidates.append({
                            'token': tok,
                            'score': float(state.get('raw_logp', 0.0) + succ.get('raw_logp', 0.0)),
                            'raw_logp': float(state.get('raw_logp', 0.0) + succ.get('raw_logp', 0.0)),
                            'trace': list(state.get('trace', [])) + ([dict(succ['selected_edit'])] if succ.get('selected_edit', None) is not None else []),
                            'step_idx': int(step_idx + 1),
                            'chain_rank': int(state.get('chain_rank', 0)),
                            'last_key': key,
                        })

                    if not step_candidates:
                        break

                    executed_iters = int(step_idx + 1)
                    eval_in = [(float(c['score']), c['token']) for c in step_candidates]
                    step_eval_res = _run_bfgs_on_hyps(
                        eval_in,
                        seed_offset=20_000 + int(step_idx) * 1_000,
                    )
                    eval_by_key: Dict[Tuple[int, ...], Dict[str, Any]] = {}
                    for pred_expr, loss_val, r2_val, tok in zip(
                            step_eval_res.get('all_bfgs_preds', []),
                            step_eval_res.get('all_bfgs_loss', []),
                            step_eval_res.get('all_bfgs_r2', []),
                            step_eval_res.get('all_tokens', [])):
                        if tok is None:
                            continue
                        eval_by_key[_token_to_key(tok)] = {
                            'pred_expr': pred_expr,
                            'bfgs_loss': float(loss_val),
                            'bfgs_r2': float(r2_val),
                            'token': tok,
                        }

                    for cand in step_candidates:
                        key = cand.get('last_key')
                        eval_info = eval_by_key.get(key)
                        if eval_info is None:
                            continue
                        merged = dict(cand)
                        merged.update(eval_info)
                        merged['origin'] = f"repair_chain_{int(merged.get('chain_rank', 0))}"
                        prev = all_records.get(key)
                        if prev is None or _better_record(prev, merged):
                            all_records[key] = merged
                        if _better_record(best_record, merged):
                            best_record = merged
                        next_active_states.append({
                            'tokens': merged['token'].to(cfg_params.device).unsqueeze(0),
                            'score': float(merged.get('score', 0.0)),
                            'raw_logp': float(merged.get('raw_logp', 0.0)),
                            'trace': list(merged.get('trace', [])),
                            'chain_rank': int(merged.get('chain_rank', 0)),
                            'last_key': key,
                        })

                    if best_record is not None:
                        best_mse_now = float(best_record.get('bfgs_loss', float('nan')))
                        if np.isfinite(best_mse_now) and best_mse_now <= float(repair_stop_mse):
                            return {
                                'records': list(all_records.values()),
                                'best_record': best_record,
                                'early_stop': True,
                                'executed_iters': int(executed_iters),
                            }

                    active_states = next_active_states
                    if not active_states:
                        break

                return {
                    'records': list(all_records.values()),
                    'best_record': best_record,
                    'early_stop': False,
                    'executed_iters': int(executed_iters),
                }
            baseline_t0 = time.perf_counter()
            baseline_res = _run_bfgs_on_hyps(valid_hyps_baseline, seed_offset=10_000)
            baseline_elapsed = float(time.perf_counter() - baseline_t0)
            baseline_best_r2 = float(baseline_res.get('best_bfgs_r2', [float('nan')])[0])

            repair_seed_topk = int(max(1, int(getattr(cfg_params, 'repair_seed_topk', min(8, max(1, len(valid_hyps_baseline)))))))
            repair_seed_hyps: List[Tuple[float, Any]] = list(valid_hyps_baseline[:repair_seed_topk])
            repair_seed_res = _empty_eval_result(token_fallback=None)
            repair_elapsed = 0.0
            repair_executed_iters = 0
            repair_best_edit_steps = 0
            repair_t0 = time.perf_counter()
            if repair_seed_hyps:
                repair_seed_res = _run_bfgs_on_hyps(repair_seed_hyps, seed_offset=15_000)

            repair_best_origin = "beam"
            repair_only_res = None
            repair_guided_records: List[Dict[str, Any]] = []
            repair_early_stop_hit = False
            skip_repair_due_to_perfect_baseline = bool(
                use_repair and np.isfinite(baseline_best_r2) and (baseline_best_r2 >= float(repair_skip_baseline_r2) - 1.0e-12)
            )

            if use_repair and skip_repair_due_to_perfect_baseline:
                repair_best_origin = "beam_skip_perfect"

            if use_repair and (not skip_repair_due_to_perfect_baseline) and self.repair_enable and (self._repair_helper is not None) and len(repair_seed_hyps) > 0 and bfgs_enabled:
                guided = _guided_repair_search_greedy_chains(repair_seed_hyps)
                repair_guided_records = list(guided.get('records', []))
                if repair_guided_records:
                    repair_only_res = _summarize_eval_records(repair_guided_records)
                repair_executed_iters = int(guided.get('executed_iters', 0))
                if bool(guided.get('early_stop', False)):
                    repair_early_stop_hit = True
                    repair_best_origin = 'repair_early_stop'

            repair_elapsed = float(time.perf_counter() - repair_t0) if use_repair else 0.0

            # Union = baseline beams + repair-greedy candidates.
            repair_union_res = baseline_res
            if use_repair and (repair_only_res is not None):
                union_records: List[Dict[str, Any]] = []
                for pred_expr, loss_val, r2_val, tok in zip(
                        baseline_res.get('all_bfgs_preds', []),
                        baseline_res.get('all_bfgs_loss', []),
                        baseline_res.get('all_bfgs_r2', []),
                        baseline_res.get('all_tokens', [])):
                    union_records.append({
                        'pred_expr': pred_expr,
                        'bfgs_loss': float(loss_val),
                        'bfgs_r2': float(r2_val),
                        'token': tok,
                        'score': 0.0,
                        'origin': 'beam',
                    })
                for rec in repair_guided_records:
                    merged = dict(rec)
                    merged.setdefault('origin', 'repair_greedy')
                    union_records.append(merged)

                repair_union_res = _summarize_eval_records(union_records)
                best_tok_key = _token_to_key(repair_union_res.get('best_token', [None])[0])
                repair_best_origin = 'beam'
                for rec in union_records:
                    if _token_to_key(rec.get('token', None)) == best_tok_key:
                        repair_best_origin = str(rec.get('origin', 'beam'))
                        break

                if repair_best_origin not in ('beam', 'beam_skip_perfect'):
                    for rec in repair_guided_records:
                        if _token_to_key(rec.get('token', None)) == best_tok_key:
                            repair_best_edit_steps = int(max(
                                int(rec.get('step_idx', 0)),
                                len(rec.get('trace', [])),
                            ))
                            break

            # Choose what to return as the "main" result (backward compatible)
            final_res = repair_union_res if use_repair else baseline_res
            output = {
                'pred_target': generated_hyps.hyp[0][1] if generated_hyps.hyp else [],
                'all_bfgs_preds': final_res['all_bfgs_preds'],
                'all_bfgs_loss': final_res['all_bfgs_loss'],
                'all_bfgs_r2': final_res.get('all_bfgs_r2', [float('nan')]),
                'best_bfgs_preds': final_res['best_bfgs_preds'],
                'best_bfgs_loss': final_res['best_bfgs_loss'],
                'best_bfgs_r2': final_res.get('best_bfgs_r2', [float('nan')]),
                'best_token': final_res['best_token'],
            }

            if return_baseline_and_repair:
                output.update({
                    'baseline_all_bfgs_preds': baseline_res['all_bfgs_preds'],
                    'baseline_all_bfgs_loss': baseline_res['all_bfgs_loss'],
                    'baseline_all_bfgs_r2': baseline_res.get('all_bfgs_r2', [float('nan')]),
                    'baseline_best_bfgs_preds': baseline_res['best_bfgs_preds'],
                    'baseline_best_bfgs_loss': baseline_res['best_bfgs_loss'],
                    'baseline_best_bfgs_r2': baseline_res.get('best_bfgs_r2', [float('nan')]),
                    'baseline_best_token': baseline_res['best_token'],
                    'repair_all_bfgs_preds': repair_union_res.get('all_bfgs_preds', [None]),
                    'repair_all_bfgs_loss': repair_union_res.get('all_bfgs_loss', [float('nan')]),
                    'repair_all_bfgs_r2': repair_union_res.get('all_bfgs_r2', [float('nan')]),
                    'repair_best_bfgs_preds': repair_union_res.get('best_bfgs_preds', [None]),
                    'repair_best_bfgs_loss': repair_union_res.get('best_bfgs_loss', [float('nan')]),
                    'repair_best_bfgs_r2': repair_union_res.get('best_bfgs_r2', [float('nan')]),
                    'repair_best_token': repair_union_res.get('best_token', [None]),
                    'repair_best_origin': repair_best_origin,
                    'baseline_elapsed': float(baseline_elapsed),
                    'repair_elapsed': float(repair_elapsed),
                    'total_elapsed': float(baseline_elapsed + repair_elapsed),
                    'repair_search_executed_iters': int(repair_executed_iters),
                    'repair_best_edit_steps': int(repair_best_edit_steps),
                    'repair_only_count': (len(repair_guided_records) if use_repair else 0),
                    'repair_early_stop_hit': bool(repair_early_stop_hit),
                    'repair_skipped_due_to_perfect_baseline': bool(skip_repair_due_to_perfect_baseline),
                    'repair_input_token': [repair_seed_hyps[0][1]] if repair_seed_hyps else [None],
                    'repair_input_tokens': [tok for _, tok in repair_seed_hyps],
                    'repair_input_all_bfgs_preds': repair_seed_res.get('all_bfgs_preds', [None]),
                    'repair_input_all_bfgs_loss': repair_seed_res.get('all_bfgs_loss', [float('nan')]),
                    'repair_input_all_bfgs_r2': repair_seed_res.get('all_bfgs_r2', [float('nan')]),
                    'repair_input_best_bfgs_preds': repair_seed_res.get('best_bfgs_preds', [None]),
                    'repair_input_best_bfgs_loss': repair_seed_res.get('best_bfgs_loss', [float('nan')]),
                    'repair_input_best_bfgs_r2': repair_seed_res.get('best_bfgs_r2', [float('nan')]),
                })

                # Debug: expose the chosen token sequences for baseline vs repair
                try:
                    base_tok = output.get('baseline_best_token', [None])[0]
                    rep_tok = output.get('repair_best_token', [None])[0]
                    print('baseline_best_token =', base_tok)
                    print('repair_best_token   =', rep_tok,
                          '(origin=%s)' % str(output.get('repair_best_origin', 'beam')))
                    if self.id2word is not None and base_tok is not None:
                        try:
                            base_expr = ' '.join([str(self.id2word.get(int(t), int(t))) for t in base_tok])
                            print('baseline_best_expr  =', base_expr)
                        except Exception:
                            pass
                    if self.id2word is not None and rep_tok is not None:
                        try:
                            rep_expr = ' '.join([str(self.id2word.get(int(t), int(t))) for t in rep_tok])
                            print('repair_best_expr    =', rep_expr)
                        except Exception:
                            pass
                except Exception:
                    pass

            self.eq = output['best_bfgs_preds']
            return output

    def _analyze_prefix_tree_context(self, seq, arity_1_ids, arity_2_ids, transcendental_ids, pow_id, c_id, start_id=1):
        """Analyze next-token constraints using the centralized vocab-driven grammar helper.

        `arity_1_ids` / `arity_2_ids` are kept for interface compatibility with existing call sites, but
        when a PrefixRepairHelper is available we delegate to its arity()/context logic so that all
        scripts share the same syntax-validity source of truth.
        """
        helper = getattr(self, "_repair_helper", None)
        if helper is None and isinstance(getattr(self, 'word2id', None), dict) and isinstance(getattr(self, 'id2word', None), dict) and self.word2id:
            helper = PrefixRepairHelper(self.word2id, self.id2word)
        if helper is not None and hasattr(helper, 'analyze_prefix_tree_context'):
            return helper.analyze_prefix_tree_context(
                seq=seq,
                transcendental_ids=transcendental_ids,
                pow_id=pow_id,
                c_id=c_id,
                start_id=start_id,
            )

        # Conservative fallback if helper is unavailable.
        stack = [[None, 1, set()]]
        start_idx = 1 if len(seq) > 0 and seq[0] == start_id else 0
        for token in seq[start_idx:]:
            if len(stack) == 0:
                break
            stack[-1][1] -= 1
            current_inherited_constraints = stack[-1][2].copy()
            if c_id is not None and len(stack) > 0 and stack[-1][0] == pow_id and stack[-1][1] == 0:
                current_inherited_constraints.add(c_id)
            new_constraints_for_children = current_inherited_constraints.copy()
            if token in transcendental_ids:
                new_constraints_for_children.update(transcendental_ids)
            if pow_id is not None and token == pow_id:
                new_constraints_for_children.add(pow_id)
            if token in arity_2_ids:
                stack.append([token, 2, new_constraints_for_children])
            elif token in arity_1_ids:
                stack.append([token, 1, new_constraints_for_children])
            while len(stack) > 0 and stack[-1][1] == 0:
                stack.pop()
        valency = sum([s[1] for s in stack])
        current_forbidden_set = stack[-1][2].copy() if len(stack) > 0 else set()
        if c_id is not None and len(stack) > 0 and stack[-1][0] == pow_id and stack[-1][1] == 1:
            current_forbidden_set.add(c_id)
        return valency, current_forbidden_set



EditSR = Model
