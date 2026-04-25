import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# -----------------------------------------------------------------------------
# Vocab-driven prefix-grammar helper
# -----------------------------------------------------------------------------

UNARY_OP_NAMES = {
    # trig / hyperbolic
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    # exp/log
    "exp", "ln", "log",
    # misc
    "sqrt", "abs",
}

BINARY_OP_NAMES = {
    "add", "sub", "mul", "div", "pow",
}

CONST_NAMES = {"c", "pi", "e", "E"}
OP_ARITY_BY_NAME = {**{name: 1 for name in UNARY_OP_NAMES}, **{name: 2 for name in BINARY_OP_NAMES}}
VAR_NAME_RE = re.compile(r"^x_(\d+)$")


def _is_int_name(name: str) -> bool:
    s = name.strip()
    if not s:
        return False
    if s[0] == "-":
        s = s[1:]
    return s.isdigit()


@dataclass
class VocabIds:
    pad_id: int
    start_id: int
    finish_id: int


@dataclass
class RootCorruptionPair:
    """A single forward corruption step record for root-level Tagger training.

    Fields are all in *body index space* (no S/F/PAD).
    - prev_body: previous state before the sampled edit
    - cur_body: current state after the sampled edit
    - forward_op: name of the forward corruption op applied to prev to get cur
    - root_idx: edited root position in prev/cur (body index), or -1 for identity
    - prev_span: (i,j) span of the edited subtree in prev_body
    - cur_span: (i,j) span of the edited subtree in cur_body
    - prev_subtree: the subtree tokens from prev_body at prev_span (used as target for reverse rewrite)
    - prev_root_token: the root token from prev_body at root_idx (used as target for OP_REPLACE/DELETE_SUBTREE reverse actions)
    """
    prev_body: List[int]
    cur_body: List[int]
    step_idx: int
    forward_op: str
    root_idx: int
    prev_span: Tuple[int, int]
    cur_span: Tuple[int, int]
    prev_subtree: List[int]
    prev_root_token: Optional[int]


class PrefixRepairHelper:
    """Prefix expression utilities.

    Conventions:
      - full sequence: [S] <body tokens> [F] PAD...
      - body is prefix-notation expression (no S/F/PAD).

    IMPORTANT: This helper does NOT provide any "projection-to-valid" repair.
    All corruption/decoding routines are designed to be grammar-preserving.
    """

    def __init__(self, word2id: Dict[str, int], id2word: Dict[int, str]):
        self.word2id = dict(word2id)
        self.id2word = dict(id2word)

        # Extra special/sentinel tokens (e.g. <repl:0>, <insert:0>) are allowed in the vocab for conditioning,
        # but MUST NOT be treated as grammar leaves (otherwise corruption/decoding may sample them).
        self.extra_special_tokens = {w for w in self.word2id.keys() if isinstance(w, str) and w.startswith("<") and w.endswith(">")}
        self.extra_special_ids = {int(self.word2id[w]) for w in self.extra_special_tokens if w in self.word2id}

        pad_id = int(self.word2id.get("P", 0))
        start_id = int(self.word2id.get("S", 1))
        finish_id = int(self.word2id.get("F", 2))
        self.ids = VocabIds(pad_id=pad_id, start_id=start_id, finish_id=finish_id)

        # Operator ids
        self.unary_ids: List[int] = [int(self.word2id[n]) for n, ar in OP_ARITY_BY_NAME.items() if ar == 1 and n in self.word2id]
        self.binary_ids: List[int] = [int(self.word2id[n]) for n, ar in OP_ARITY_BY_NAME.items() if ar == 2 and n in self.word2id]
        self.op_ids: List[int] = sorted(set(self.unary_ids + self.binary_ids))

        # Leaf ids: variables + constants + integer literals
        self.var_ids: List[int] = []
        self.const_leaf_ids: List[int] = []
        for w, i in self.word2id.items():
            if isinstance(w, str) and VAR_NAME_RE.match(w):
                if int(i) not in self.extra_special_ids:
                    self.var_ids.append(int(i))

        # The project uses a unified 1-indexed variable protocol: x_1..x_n.
        self.var_index_base = 1
        # Optional forward-op sampling weights for synthetic corruption. The model may
        # overwrite this after helper construction to better balance rare reverse actions.
        self.corruption_action_weights: Dict[str, float] = {}
        for w in CONST_NAMES:
            if w in self.word2id:
                if int(self.word2id[w]) not in self.extra_special_ids:
                    self.const_leaf_ids.append(int(self.word2id[w]))
        for w, i in self.word2id.items():
            if _is_int_name(w):
                if int(i) not in self.extra_special_ids:
                    self.const_leaf_ids.append(int(i))

        # Fallback leaf ids if needed
        self.default_leaf_ids: List[int] = []
        for w, i in self.word2id.items():
            if i in (self.ids.pad_id, self.ids.start_id, self.ids.finish_id):
                continue
            if i in self.op_ids:
                continue
            if int(i) in self.extra_special_ids:
                continue
            self.default_leaf_ids.append(int(i))
        if not self.default_leaf_ids:
            # extreme fallback
            self.default_leaf_ids = [int(self.ids.start_id)]

        # Small same-arity swap groups (optional)
        self._unary_swap_pairs: List[Tuple[int, int]] = []
        for a, b in [("sin", "cos"), ("asin", "acos"), ("ln", "exp"), ("log", "exp")]:
            if a in self.word2id and b in self.word2id:
                self._unary_swap_pairs.append((int(self.word2id[a]), int(self.word2id[b])))
        self._binary_swap_pairs: List[Tuple[int, int]] = []
        for a, b in [("add", "sub"), ("mul", "div")]:
            if a in self.word2id and b in self.word2id:
                self._binary_swap_pairs.append((int(self.word2id[a]), int(self.word2id[b])))

    # -------------------------- basic utilities --------------------------

    @property
    def pad_id(self) -> int:
        return self.ids.pad_id

    @property
    def start_id(self) -> int:
        return self.ids.start_id

    @property
    def finish_id(self) -> int:
        return self.ids.finish_id

    def truncate_after_F_inplace(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return a clone with everything after first F set to PAD (per row)."""
        if tokens.dim() != 2:
            raise ValueError("tokens must be [B, L]")
        out = tokens.clone()
        B, L = out.shape
        for b in range(B):
            seq = out[b].tolist()
            if self.finish_id in seq:
                f = seq.index(self.finish_id)
                if f + 1 < L:
                    out[b, f + 1:] = self.pad_id
        return out

    def extract_body(self, seq: Sequence[int]) -> List[int]:
        """Extract body (between S and F), removing PAD."""
        seq2 = [int(t) for t in seq if int(t) != self.pad_id]
        if seq2 and seq2[0] == self.start_id:
            seq2 = seq2[1:]
        if self.finish_id in seq2:
            seq2 = seq2[: seq2.index(self.finish_id)]
        return seq2

    def pack(self, body: List[int], max_len: int) -> List[int]:
        seq = [self.start_id] + [int(t) for t in body] + [self.finish_id]
        if len(seq) < max_len:
            seq += [self.pad_id] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
            if self.finish_id not in seq:
                seq[-1] = self.finish_id
        return seq

    def arity(self, tok: int) -> int:
        tok = int(tok)
        # Extra sentinel/tag tokens (e.g. <repl:0>, <insert:0>) are allowed in the vocab
        # for conditioning, but they are NOT grammar symbols. Treat them as invalid.
        if tok in getattr(self, "extra_special_ids", set()):
            return -1
        # Core specials are also invalid inside bodies.
        if tok in (int(self.ids.pad_id), int(self.ids.start_id), int(self.ids.finish_id)):
            return -1
        if tok in self.binary_ids:
            return 2
        if tok in self.unary_ids:
            return 1
        # Everything else is treated as a leaf.
        return 0

    def validate_body(self, body: Sequence[int]) -> bool:
        need = 1
        for t in body:
            if need <= 0:
                return False
            need -= 1
            need += self.arity(int(t))
        return need == 0

    def subtree_end(self, body: Sequence[int], i: int) -> int:
        need = 1
        j = int(i)
        n = len(body)
        while j < n and need > 0:
            t = int(body[j])
            need -= 1
            need += self.arity(t)
            j += 1
        return j

    def all_subtree_spans(self, body: Sequence[int]) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        for i in range(len(body)):
            spans.append((i, self.subtree_end(body, i)))
        return spans

    # -------------------------- leaf pools --------------------------

    def global_leaf_ids(self) -> List[int]:
        """Return the unrestricted grammar leaf pool implied by the current vocab.

        This is the single source of truth for leaf sampling/decoding across training and inference:
        every non-special, non-operator vocab item is treated as a legal leaf. Variable ids are
        discovered directly from the current vocab (for example, if x_7/x_8 are absent, they are
        simply absent from the pool).
        """
        return list(self.default_leaf_ids)

    def allowed_leaf_ids_from_body(self, body: Sequence[int]) -> List[int]:
        """Return the unrestricted grammar leaf pool.

        The previous implementation restricted variables to those already present in `body`. The
        current project intentionally removes that restriction so that training-time corruption,
        free-run initialization, and inference-time repair all operate on the same leaf space.
        """
        _ = body
        return self.global_leaf_ids()

    def _random_const_leaf(self) -> int:
        if self.const_leaf_ids:
            # Prefer '1' if present
            one_id = self.word2id.get("1", None)
            if one_id is not None:
                return int(one_id)
            return int(random.choice(self.const_leaf_ids))
        # last resort: any leaf
        return int(random.choice(self.default_leaf_ids))

    def random_leaf(self, allowed_leaf_ids: Sequence[int]) -> int:
        if allowed_leaf_ids:
            return int(random.choice(list(allowed_leaf_ids)))
        return self._random_const_leaf()

    def random_subtree(
            self,
            allowed_leaf_ids: Sequence[int],
            max_nodes: int = 9,
            max_depth: int = 4,
            depth: int = 0,
    ) -> List[int]:
        """Generate a random valid prefix subtree using ONLY allowed leaves."""
        if depth >= max_depth or max_nodes <= 1 or random.random() < 0.35:
            return [self.random_leaf(allowed_leaf_ids)]

        choose_binary = bool(self.binary_ids) and (random.random() < 0.6)
        if choose_binary:
            op = int(random.choice(self.binary_ids))
            left = self.random_subtree(allowed_leaf_ids, max_nodes=max_nodes - 1, max_depth=max_depth, depth=depth + 1)
            right = self.random_subtree(
                allowed_leaf_ids,
                max_nodes=max(1, max_nodes - 1 - len(left)),
                max_depth=max_depth,
                depth=depth + 1,
            )
            return [op] + left + right

        # unary
        if self.unary_ids:
            op = int(random.choice(self.unary_ids))
            child = self.random_subtree(allowed_leaf_ids, max_nodes=max_nodes - 1, max_depth=max_depth, depth=depth + 1)
            return [op] + child

        return [self.random_leaf(allowed_leaf_ids)]

    # -------------------------- corruption ops (grammar-preserving) --------------------------

    def corrupt_body(self, body: Sequence[int], strength: float, max_body_len: int) -> List[int]:
        """Grammar-preserving corruption.

        - NEVER calls any "make valid" projection.
        - If `body` is not valid, returns it unchanged.
        - Applies a random mixture of subtree add/delete/replace and same-arity op replace.
        """
        body = [int(t) for t in body]
        if not body or not self.validate_body(body):
            return list(body)

        allowed_leaf = self.allowed_leaf_ids_from_body(body)

        # Randomize intensity even at fixed strength
        s = float(max(0.0, min(1.0, strength)))
        # edits range: 0..(1 + 7*s), then randomized
        max_edits = int(round(1 + 7 * s))
        n_edits = random.randint(0, max_edits)
        if n_edits <= 0:
            return list(body)

        cur = list(body)
        for _ in range(n_edits):
            if not cur or not self.validate_body(cur):
                break

            r = random.random()
            # Stronger noise => more structure ops
            if r < 0.30:
                cand = self._op_operator_replace(cur)
            elif r < 0.60:
                cand = self._op_subtree_delete(cur, allowed_leaf)
            elif r < 0.85:
                cand = self._op_subtree_add(cur, allowed_leaf)
            else:
                cand = self._op_subtree_replace(cur, allowed_leaf)

            if cand is None:
                continue
            if len(cand) > max_body_len:
                # keep grammar by applying a delete instead of truncation
                cand2 = self._op_subtree_delete(cand, allowed_leaf)
                cand = cand2 if (cand2 is not None and len(cand2) <= max_body_len) else None
            if cand is None:
                continue
            if self.validate_body(cand) and len(cand) <= max_body_len:
                cur = cand

        return cur

    def _op_subtree_delete(self, body: Sequence[int], allowed_leaf: Sequence[int]) -> Optional[List[int]]:
        spans = self.all_subtree_spans(body)
        if not spans:
            return None
        # Prefer non-leaf subtrees
        non_leaf = [(i, j) for (i, j) in spans if (j - i) >= 2]
        cand = non_leaf if non_leaf else spans
        i, j = random.choice(cand)
        # Use a random constant OR a variable that already appears in the original GT/body.
        # `allowed_leaf` is constructed as: constants + vars_in_body, so this never introduces x_5 when GT only used x_1..x_3.
        leaf = int(random.choice(list(allowed_leaf))) if allowed_leaf else self._random_const_leaf()
        return list(body[:i]) + [leaf] + list(body[j:])

    def _op_subtree_add(self, body: Sequence[int], allowed_leaf: Sequence[int]) -> Optional[List[int]]:
        leaf_pos = [i for i, t in enumerate(body) if self.arity(int(t)) == 0]
        if not leaf_pos:
            return None
        i = random.choice(leaf_pos)
        subtree = self.random_subtree(allowed_leaf, max_nodes=9, max_depth=4)
        return list(body[:i]) + subtree + list(body[i + 1:])

    def _op_subtree_replace(self, body: Sequence[int], allowed_leaf: Sequence[int]) -> Optional[List[int]]:
        spans = self.all_subtree_spans(body)
        if not spans:
            return None
        i, j = random.choice(spans)
        # replacement subtree size roughly proportional to removed span
        span_len = max(1, j - i)
        max_nodes = min(11, max(3, span_len + random.randint(-2, 2)))
        subtree = self.random_subtree(allowed_leaf, max_nodes=max_nodes, max_depth=5)
        return list(body[:i]) + subtree + list(body[j:])

    def _op_operator_replace(self, body: Sequence[int]) -> Optional[List[int]]:
        op_pos = [i for i, t in enumerate(body) if int(t) in self.op_ids]
        if not op_pos:
            return None
        i = random.choice(op_pos)
        t = int(body[i])
        a = self.arity(t)

        out = list(body)
        if a == 1 and self.unary_ids:
            for x, y in self._unary_swap_pairs:
                if t == x:
                    out[i] = y
                    return out
                if t == y:
                    out[i] = x
                    return out
            choices = [u for u in self.unary_ids if u != t]
            if choices:
                out[i] = int(random.choice(choices))
                return out
        if a == 2 and self.binary_ids:
            for x, y in self._binary_swap_pairs:
                if t == x:
                    out[i] = y
                    return out
                if t == y:
                    out[i] = x
                    return out
            choices = [u for u in self.binary_ids if u != t]
            if choices:
                out[i] = int(random.choice(choices))
                return out
        return None

    # -------------------------- traceable multi-step corruption (x_{t-1} -> x_t) --------------------------

    def allowed_leaf_ids_from_nvars(self, n_vars: int) -> List[int]:
        """Return leaf ids consistent with the active variable count.

        Variables follow the helper's unified protocol x_1..x_n. Constants remain always available.
        When n_vars is None/<=0, this falls back to the full grammar leaf pool.
        """
        try:
            n_vars = int(n_vars)
        except Exception:
            n_vars = 0
        if n_vars <= 0:
            return self.global_leaf_ids()
        out: List[int] = []
        var_min = int(getattr(self, 'var_index_base', 1))
        var_max = int(var_min + n_vars - 1)
        for vid in self.var_ids:
            w = str(self.id2word.get(int(vid), ''))
            m = VAR_NAME_RE.match(w)
            if m is None:
                continue
            idx = int(m.group(1))
            if var_min <= idx <= var_max:
                out.append(int(vid))
        out.extend(int(x) for x in self.const_leaf_ids)
        out = sorted(set(int(x) for x in out))
        return out if out else self.global_leaf_ids()

    def _apply_one_corruption_step(
            self,
            prev_body: Sequence[int],
            allowed_leaf: Sequence[int],
            max_body_len: int,
            max_insert: int,
            max_tries: int = 20,
    ) -> Tuple[List[int], List[Optional[int]]]:
        """Apply ONE forward corruption step with explicit provenance.

        Returns:
          cur_body: new body tokens
          provenance: list where provenance[p] is index in prev_body that cur_body[p] came from, or None if inserted

        Design constraints:
          - Output must remain prefix-valid (validate_body).
          - Length is bounded by max_body_len.
          - The reverse of this step must be representable with per-token INSERT slots of size <= max_insert.
            Concretely, any deleted contiguous span between two preserved tokens in prev_body must be <= max_insert.
        """
        prev = [int(t) for t in prev_body]
        if not prev or (not self.validate_body(prev)):
            prov0: List[Optional[int]] = [i for i in range(len(prev))]
            return list(prev), prov0

        max_body_len = int(max(1, max_body_len))
        max_insert = int(max(0, max_insert))

        spans = self.all_subtree_spans(prev)

        def _no_op():
            prov = [i for i in range(len(prev))]
            return list(prev), [int(x) for x in prov]

        # try a few times to satisfy representability constraints
        for _ in range(int(max_tries)):
            r = random.random()
            if r < 0.25:
                op = "op_replace"
            elif r < 0.50:
                op = "subtree_delete"
            elif r < 0.75:
                op = "subtree_add"
            else:
                op = "subtree_replace"

            # ---------- operator replace ----------
            if op == "op_replace":
                op_pos = [i for i, t in enumerate(prev) if int(t) in self.op_ids]
                if not op_pos:
                    continue
                i = random.choice(op_pos)
                t = int(prev[i])
                a = self.arity(t)
                cur = list(prev)
                if a == 1:
                    choices = [u for u in self.unary_ids if u != t]
                    if choices:
                        cur[i] = int(random.choice(choices))
                elif a == 2:
                    choices = [u for u in self.binary_ids if u != t]
                    if choices:
                        cur[i] = int(random.choice(choices))
                if len(cur) <= max_body_len and self.validate_body(cur):
                    prov = [j for j in range(len(cur))]
                    return cur, [int(x) for x in prov]
                continue

            # Need subtree spans for structure ops
            if not spans:
                continue

            # Prefer non-leaf spans
            non_leaf = [(i, j) for (i, j) in spans if (j - i) >= 2]
            span_choices = non_leaf if non_leaf else spans

            i, j = random.choice(span_choices)

            # deleted tokens between preserved prev indices i and j:
            # when we delete/replace a subtree at [i:j), the root prev[i] remains as an anchor,
            # and the missing tail prev[i+1:j) must be inserted back in reverse.
            missing_len = max(0, (j - i - 1))

            # ---------- subtree delete ----------
            if op == "subtree_delete":
                if missing_len > max_insert:
                    continue
                leaf = int(random.choice(list(allowed_leaf))) if allowed_leaf else self._random_const_leaf()
                cur = list(prev[:i]) + [leaf] + list(prev[j:])

                if len(cur) > max_body_len or (not self.validate_body(cur)):
                    continue

                # provenance mapping
                prov: List[Optional[int]] = []
                for p in range(i):
                    prov.append(p)
                prov.append(i)  # replacement leaf maps to old subtree root
                removed = (j - i)
                for p in range(i + 1, len(cur)):
                    prov.append(p + removed - 1)
                return cur, prov

            # ---------- subtree add ----------
            if op == "subtree_add":
                leaf_pos = [p for p, t in enumerate(prev) if self.arity(int(t)) == 0]
                if not leaf_pos:
                    continue
                i = random.choice(leaf_pos)
                subtree = self.random_subtree(allowed_leaf, max_nodes=9, max_depth=4)
                cur = list(prev[:i]) + list(subtree) + list(prev[i + 1:])
                if len(cur) > max_body_len or (not self.validate_body(cur)):
                    continue

                prov: List[Optional[int]] = []
                for p in range(i):
                    prov.append(p)
                prov.append(i)  # new subtree root maps to old leaf
                for _ in range(len(subtree) - 1):
                    prov.append(None)
                for p in range(i + len(subtree), len(cur)):
                    prov.append(p - (len(subtree) - 1))
                return cur, prov

            # ---------- subtree replace ----------
            if op == "subtree_replace":
                # Legacy path: keep rewrite restricted to operator-rooted spans only.
                if self.arity(int(prev[i])) == 0:
                    continue
                if missing_len > max_insert:
                    continue
                span_len = max(1, j - i)
                max_nodes = min(11, max(3, span_len + random.randint(-2, 2)))
                subtree = self.random_subtree(allowed_leaf, max_nodes=max_nodes, max_depth=5)
                cur = list(prev[:i]) + list(subtree) + list(prev[j:])
                if len(cur) > max_body_len or (not self.validate_body(cur)):
                    continue

                prov: List[Optional[int]] = []
                for p in range(i):
                    prov.append(p)
                prov.append(i)  # new root maps to old root
                for _ in range(len(subtree) - 1):
                    prov.append(None)
                delta = len(subtree) - (j - i)
                for p in range(i + len(subtree), len(cur)):
                    prov.append(p - delta)
                return cur, prov

        # fallback: no-op
        prov = [i for i in range(len(prev))]
        return list(prev), [int(x) for x in prov]

    def sample_corruption_pair(
            self,
            gt_body: Sequence[int],
            step_idx: int,
            max_body_len: int,
            max_insert: int,
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[List[int]]]:
        """Create a traceable pair (prev, cur) and token-level edit labels for cur -> prev.

        We train *one-step reverse*: given the current corrupted state, predict edits that transform it into the previous state.
        This avoids ambiguous labels under multi-step corruption.

        step_idx:
          - step_idx == 0: identity sample (prev=cur=gt), all KEEP, no inserts.
          - step_idx >= 1: apply (step_idx-1) forward edits to get the previous state, then one more step to get the current corrupted state.

        Returns:
          prev_body, cur_body,
          op_tags (len(cur_body), 0=KEEP,1=DELETE,2=REPLACE),
          ins_counts (len(cur_body), 0..max_insert),
          replace_targets (len(cur_body), token ids; ignored when not REPLACE),
          insert_targets (len(cur_body), list[max_insert] token ids; PAD when unused)
        """
        body0 = [int(t) for t in gt_body]
        if not body0 or (not self.validate_body(body0)):
            return list(body0), list(body0), [], [], [], []

        allowed_leaf = self.allowed_leaf_ids_from_body(body0)
        max_body_len = int(max(1, max_body_len))
        max_insert = int(max(0, max_insert))
        step_idx = int(max(0, min(step_idx, 10_000)))

        # identity sample at step=0 (trains the model's "do nothing" behavior)
        if step_idx == 0:
            cur = list(body0)[:max_body_len]
            prev = list(cur)
            KEEP = 0
            op_tags = [KEEP for _ in cur]
            ins_counts = [0 for _ in cur]
            replace_targets = [int(self.pad_id) for _ in cur]
            insert_targets = [[int(self.pad_id)] * int(max_insert) for _ in cur]
            return prev, cur, op_tags, ins_counts, replace_targets, insert_targets

        prev = list(body0)
        # apply step_idx-1 forward edits to get the previous state
        for _ in range(max(0, step_idx - 1)):
            prev2, _ = self._apply_one_corruption_step(prev, allowed_leaf, max_body_len, max_insert)
            prev = prev2

        cur, prov = self._apply_one_corruption_step(prev, allowed_leaf, max_body_len, max_insert)

        KEEP, DELETE, REPLACE = 0, 1, 2
        op_tags: List[int] = [KEEP for _ in cur]
        ins_counts: List[int] = [0 for _ in cur]
        replace_targets: List[int] = [int(self.pad_id) for _ in cur]
        insert_targets: List[List[int]] = [[int(self.pad_id)] * int(max_insert) for _ in cur]

        # 1) op tags + replace targets
        for p, tok in enumerate(cur):
            pk = prov[p]
            if pk is None:
                op_tags[p] = DELETE
                continue
            pk = int(pk)
            if pk < 0 or pk >= len(prev):
                op_tags[p] = DELETE
                continue
            tgt_tok = int(prev[pk])
            if int(tok) == tgt_tok:
                op_tags[p] = KEEP
            else:
                op_tags[p] = REPLACE
                replace_targets[p] = tgt_tok

        # 2) insertion plan: fill missing segments in prev that are not mapped from cur.
        #    For each preserved prev index k, insert prev[k+1 : next_k] after its corresponding cur token.
        prev_to_cur: Dict[int, int] = {}
        for p, pk in enumerate(prov):
            if pk is None:
                continue
            prev_to_cur[int(pk)] = int(p)

        if prev_to_cur:
            sorted_prev = sorted(prev_to_cur.keys())
            # defensive: ensure we have an anchor for root
            if 0 not in prev_to_cur:
                # anchor root to first mapped token
                prev_to_cur[0] = prev_to_cur[sorted_prev[0]]
                sorted_prev = sorted(prev_to_cur.keys())

            for idx_k, k in enumerate(sorted_prev):
                cur_pos = prev_to_cur[k]
                next_k = sorted_prev[idx_k + 1] if (idx_k + 1) < len(sorted_prev) else len(prev)
                if next_k <= k + 1:
                    continue
                missing = [int(x) for x in prev[k + 1: next_k]]
                if not missing:
                    continue
                # representability should be guaranteed by _apply_one_corruption_step; keep a guard anyway
                if len(missing) > max_insert and max_insert > 0:
                    missing = missing[:max_insert]

                m = min(len(missing), int(max_insert))
                ins_counts[cur_pos] = m
                for j in range(m):
                    insert_targets[cur_pos][j] = int(missing[j])

        return list(prev), list(cur), op_tags, ins_counts, replace_targets, insert_targets

    # -------------------------- root-level traceable corruption (for Tagger v2) --------------------------

    def _apply_one_corruption_step_with_record(
            self,
            prev_body: Sequence[int],
            allowed_leaf: Sequence[int],
            max_body_len: int,
            max_tries: int = 30,
            rewrite_budget: Optional[int] = None,
    ) -> Optional[RootCorruptionPair]:
        """Apply ONE grammar-preserving corruption step and return a record.

        Position-driven synthetic corruption:
          1) sample one root position uniformly at random,
          2) derive the feasible corruption actions at that position from the local arity,
          3) sample uniformly among the feasible actions,
          4) realize the chosen action with a grammar-valid edit.

        KEEP is never sampled here; it is only used for Tagger supervision.
        """
        prev = [int(t) for t in prev_body]
        if (not prev) or (not self.validate_body(prev)):
            return None

        allowed_leaf = [int(x) for x in allowed_leaf] if allowed_leaf else self.allowed_leaf_ids_from_body(prev)
        if not allowed_leaf:
            allowed_leaf = self.global_leaf_ids()
        allowed_leaf = [int(x) for x in allowed_leaf]
        max_body_len = int(max(1, max_body_len))
        rewrite_budget = None if rewrite_budget is None else int(max(2, rewrite_budget))

        spans = self.all_subtree_spans(prev)
        if not spans:
            return None
        span_by_root = {int(i): (int(i), int(j)) for (i, j) in spans}
        root_positions = [int(i) for (i, _j) in spans]

        def _distinct_leaf(tok0: int) -> Optional[int]:
            choices = [int(v) for v in allowed_leaf if int(v) != int(tok0)]
            return int(random.choice(choices)) if choices else None

        def _distinct_same_arity_op(tok0: int) -> Optional[int]:
            a = int(self.arity(int(tok0)))
            if a == 1:
                choices = [int(v) for v in self.unary_ids if int(v) != int(tok0)]
            elif a == 2:
                choices = [int(v) for v in self.binary_ids if int(v) != int(tok0)]
            else:
                choices = []
            return int(random.choice(choices)) if choices else None

        def _random_nonleaf_subtree(max_nodes: int, *, exclude: Optional[Sequence[int]] = None, preserve_leaf: Optional[int] = None) -> Optional[List[int]]:
            max_nodes = int(max(2, max_nodes))
            exclude_list = [int(x) for x in (exclude or [])]
            preserve_leaf = None if preserve_leaf is None else int(preserve_leaf)
            for _ in range(40):
                if preserve_leaf is not None:
                    subtree = None
                    if self.unary_ids and max_nodes >= 2 and random.random() < 0.5:
                        op = int(random.choice(self.unary_ids))
                        subtree = [op, preserve_leaf]
                    elif self.binary_ids and max_nodes >= 3:
                        op = int(random.choice(self.binary_ids))
                        sibling_budget = max(1, max_nodes - 2)
                        sibling = [int(x) for x in self.random_subtree(allowed_leaf, max_nodes=sibling_budget, max_depth=4)]
                        if random.random() < 0.5:
                            subtree = [op, preserve_leaf] + sibling
                        else:
                            subtree = [op] + sibling + [preserve_leaf]
                    if subtree is None:
                        continue
                else:
                    subtree = [int(x) for x in self.random_subtree(allowed_leaf, max_nodes=max_nodes, max_depth=5)]
                if len(subtree) < 2:
                    continue
                if rewrite_budget is not None and len(subtree) > int(rewrite_budget):
                    continue
                if exclude_list and subtree == exclude_list:
                    continue
                if self.validate_body(subtree):
                    return [int(x) for x in subtree]
            return None

        for _ in range(int(max_tries)):
            i = int(random.choice(root_positions))
            s, t = span_by_root[int(i)]
            prev_subtree = [int(x) for x in prev[s:t]]
            tok0 = int(prev[i])
            ar = int(self.arity(tok0))
            if ar < 0:
                continue

            feasible_ops: List[str] = []
            if ar == 0:
                # Leaf positions only admit local leaf replacement or insertion.
                # Synthetic corruption never samples a generic subtree rewrite at a leaf.
                if _distinct_leaf(tok0) is not None:
                    feasible_ops.append('leaf_replace')
                insert_budget = int(min(rewrite_budget if rewrite_budget is not None else 9, max(2, max_body_len - (len(prev) - 1))))
                if _random_nonleaf_subtree(insert_budget, preserve_leaf=tok0) is not None:
                    feasible_ops.append('leaf_insert')
            else:
                # Operator positions admit operator replacement, subtree rewrite, or subtree deletion.
                if _distinct_same_arity_op(tok0) is not None:
                    feasible_ops.append('op_replace')
                rewrite_budget_local = int(min(rewrite_budget if rewrite_budget is not None else max(2, t - s + 2), max(2, max_body_len - (len(prev) - (t - s)))))
                if _random_nonleaf_subtree(rewrite_budget_local, exclude=prev_subtree) is not None:
                    feasible_ops.append('subtree_replace')
                if _distinct_leaf(tok0) is not None or allowed_leaf:
                    feasible_ops.append('subtree_delete')

            if not feasible_ops:
                continue

            weight_map = getattr(self, 'corruption_action_weights', {}) or {}
            weights = [max(1.0e-6, float(weight_map.get(str(op_name), 1.0))) for op_name in feasible_ops]
            op = str(random.choices(feasible_ops, weights=weights, k=1)[0])

            if op == 'leaf_replace':
                leaf = _distinct_leaf(tok0)
                if leaf is None:
                    continue
                cur = list(prev)
                cur[i] = int(leaf)
                if len(cur) <= max_body_len and self.validate_body(cur):
                    return RootCorruptionPair(
                        prev_body=list(prev),
                        cur_body=list(cur),
                        step_idx=0,
                        forward_op='leaf_replace',
                        root_idx=int(i),
                        prev_span=(int(i), int(i + 1)),
                        cur_span=(int(i), int(i + 1)),
                        prev_subtree=[int(prev[i])],
                        prev_root_token=int(prev[i]),
                    )
                continue

            if op == 'op_replace':
                new_tok = _distinct_same_arity_op(tok0)
                if new_tok is None:
                    continue
                cur = list(prev)
                cur[i] = int(new_tok)
                if len(cur) <= max_body_len and self.validate_body(cur):
                    return RootCorruptionPair(
                        prev_body=list(prev),
                        cur_body=list(cur),
                        step_idx=0,
                        forward_op='op_replace',
                        root_idx=int(i),
                        prev_span=(int(i), int(i + 1)),
                        cur_span=(int(i), int(i + 1)),
                        prev_subtree=[int(prev[i])],
                        prev_root_token=int(prev[i]),
                    )
                continue

            if op == 'subtree_delete':
                leaf = _distinct_leaf(tok0)
                if leaf is None:
                    leaf = int(random.choice(allowed_leaf)) if allowed_leaf else self._random_const_leaf()
                cur = list(prev[:s]) + [int(leaf)] + list(prev[t:])
                if len(cur) <= max_body_len and self.validate_body(cur):
                    return RootCorruptionPair(
                        prev_body=list(prev),
                        cur_body=list(cur),
                        step_idx=0,
                        forward_op='subtree_delete',
                        root_idx=int(s),
                        prev_span=(int(s), int(t)),
                        cur_span=(int(s), int(s + 1)),
                        prev_subtree=[int(x) for x in prev[s:t]],
                        prev_root_token=int(prev[s]),
                    )
                continue

            if op == 'leaf_insert':
                max_nodes = int(min(rewrite_budget if rewrite_budget is not None else 9, max(2, max_body_len - (len(prev) - 1))))
                subtree = _random_nonleaf_subtree(max_nodes, preserve_leaf=tok0)
                if subtree is None:
                    continue
                cur = list(prev[:i]) + list(subtree) + list(prev[i + 1:])
                if len(cur) <= max_body_len and self.validate_body(cur):
                    return RootCorruptionPair(
                        prev_body=list(prev),
                        cur_body=list(cur),
                        step_idx=0,
                        forward_op='subtree_add',
                        root_idx=int(i),
                        prev_span=(int(i), int(i + 1)),
                        cur_span=(int(i), int(i + len(subtree))),
                        prev_subtree=[int(prev[i])],
                        prev_root_token=int(prev[i]),
                    )
                continue

            if op == 'subtree_replace':
                max_nodes = int(min(rewrite_budget if rewrite_budget is not None else max(2, (t - s) + 2), max(2, max_body_len - (len(prev) - (t - s)))))
                subtree = _random_nonleaf_subtree(max_nodes, exclude=prev_subtree)
                if subtree is None:
                    continue
                cur = list(prev[:s]) + list(subtree) + list(prev[t:])
                if len(cur) <= max_body_len and self.validate_body(cur):
                    return RootCorruptionPair(
                        prev_body=list(prev),
                        cur_body=list(cur),
                        step_idx=0,
                        forward_op='subtree_replace',
                        root_idx=int(s),
                        prev_span=(int(s), int(t)),
                        cur_span=(int(s), int(s + len(subtree))),
                        prev_subtree=[int(x) for x in prev[s:t]],
                        prev_root_token=int(prev[s]),
                    )
                continue

        return None

    def sample_root_corruption_pair(
            self,
            gt_body: Sequence[int],
            step_idx: int,
            max_body_len: int,
            T_max: int = 4,
            rewrite_budget: Optional[int] = None,
    ) -> RootCorruptionPair:
        """Generate a traceable multi-step chain and return the *last* step pair.

        If step_idx==0, returns an identity pair (prev==cur, forward_op='identity').
        """
        gt = [int(x) for x in gt_body]
        if (not gt) or (not self.validate_body(gt)):
            return RootCorruptionPair(prev_body=list(gt), cur_body=list(gt), step_idx=0, forward_op='identity',
                                      root_idx=-1,
                                      prev_span=(0, 0), cur_span=(0, 0), prev_subtree=list(gt), prev_root_token=None)

        step_idx = int(max(0, min(int(step_idx), int(T_max))))
        if step_idx == 0:
            return RootCorruptionPair(prev_body=list(gt), cur_body=list(gt), step_idx=0, forward_op='identity',
                                      root_idx=-1,
                                      prev_span=(0, 0), cur_span=(0, 0), prev_subtree=list(gt), prev_root_token=None)

        allowed_leaf = self.allowed_leaf_ids_from_body(gt)
        prev = list(gt)
        last: Optional[RootCorruptionPair] = None
        for _ in range(step_idx):
            rec = self._apply_one_corruption_step_with_record(prev, allowed_leaf=allowed_leaf,
                                                              max_body_len=max_body_len,
                                                              rewrite_budget=rewrite_budget)
            if rec is None:
                # no-op if we couldn't find a valid edit
                last = RootCorruptionPair(prev_body=list(prev), cur_body=list(prev), step_idx=0, forward_op='identity',
                                          root_idx=-1,
                                          prev_span=(0, 0), cur_span=(0, 0), prev_subtree=list(prev),
                                          prev_root_token=None)
                continue
            rec.step_idx = step_idx
            last = RootCorruptionPair(
                prev_body=list(prev),
                cur_body=list(rec.cur_body),
                step_idx=step_idx,
                forward_op=rec.forward_op,
                root_idx=rec.root_idx,
                prev_span=rec.prev_span,
                cur_span=rec.cur_span,
                prev_subtree=list(rec.prev_subtree),
                prev_root_token=rec.prev_root_token,
            )
            prev = list(rec.cur_body)

        assert last is not None
        return last

    def sample_root_corruption_chain(
            self,
            gt_body: Sequence[int],
            step_idx: int,
            max_body_len: int,
            T_max: int = 4,
            step_resample_attempts: int = 8,
            rewrite_budget: Optional[int] = None,
    ) -> List[RootCorruptionPair]:
        """Generate a traceable multi-step forward corruption chain and return *all* step records.

        This routine now targets *actual* corruption edits rather than padding the requested depth with
        identity/no-op records. We try to realize exactly ``step_idx`` random edits; if a later step becomes
        impossible under grammar / length constraints, we stop early and return the deepest reachable chain.

        Returns a list of length ``<= step_idx`` (clipped to [0, T_max]), where the s-th record corresponds to:
            prev_body = state_{s-1}, cur_body = state_s, and record.step_idx = s  (1-indexed step index)

        If step_idx==0, returns an empty list.
        """
        gt = [int(x) for x in gt_body]
        if (not gt) or (not self.validate_body(gt)):
            return []

        step_idx = int(max(0, min(int(step_idx), int(T_max))))
        if step_idx == 0:
            return []

        step_resample_attempts = int(max(1, step_resample_attempts))
        allowed_leaf = self.allowed_leaf_ids_from_body(gt)
        prev = list(gt)
        chain: List[RootCorruptionPair] = []
        for s in range(1, step_idx + 1):
            rec: Optional[RootCorruptionPair] = None
            for _ in range(step_resample_attempts):
                cand = self._apply_one_corruption_step_with_record(
                    prev,
                    allowed_leaf=allowed_leaf,
                    max_body_len=max_body_len,
                    max_tries=60,
                    rewrite_budget=rewrite_budget,
                )
                if cand is None:
                    continue
                if list(cand.cur_body) == list(prev):
                    continue
                rec = cand
                break

            if rec is None:
                # Could not realize another valid corruption step from the current body; keep the chain prefix.
                break

            rec.step_idx = int(s)
            chain.append(rec)
            prev = list(rec.cur_body)

        return chain




# -----------------------------------------------------------------------------
# Edit-guided constrained decoding (Tagger + Editor)
# -----------------------------------------------------------------------------

TAG_KEEP = 0
TAG_DELETE = 1
TAG_REPLACE = 2


def _force_logits_to_token(logits: torch.Tensor, tok_id: int) -> torch.Tensor:
    """Return a logits vector that forces selecting tok_id under argmax."""
    out = torch.full_like(logits, -1.0e9)
    out[int(tok_id)] = 0.0
    return out


def _analyze_prefix_tree_context_impl(
        helper: "PrefixRepairHelper",
        seq: Sequence[int],
        transcendental_ids: Optional[Sequence[int]] = None,
        pow_id: Optional[int] = None,
        c_id: Optional[int] = None,
        start_id: Optional[int] = None,
) -> Tuple[int, set]:
    """Centralized prefix-tree context analysis driven by the helper's arity() method."""
    transcendental_ids = set(int(x) for x in (transcendental_ids or []))
    start_id = int(helper.start_id if start_id is None else start_id)
    pow_id = None if pow_id is None else int(pow_id)
    c_id = None if c_id is None else int(c_id)

    stack = [[None, 1, set()]]
    start_idx = 1 if len(seq) > 0 and int(seq[0]) == start_id else 0

    for token in [int(t) for t in seq[start_idx:]]:
        if not stack:
            break

        stack[-1][1] -= 1
        current_inherited_constraints = set(stack[-1][2])

        if c_id is not None and pow_id is not None:
            if stack[-1][0] == pow_id and stack[-1][1] == 0:
                current_inherited_constraints.add(c_id)

        new_constraints_for_children = set(current_inherited_constraints)
        if token in transcendental_ids:
            new_constraints_for_children.update(transcendental_ids)
        if pow_id is not None and token == pow_id:
            new_constraints_for_children.add(pow_id)

        ar = int(helper.arity(token))
        if ar > 0:
            stack.append([token, ar, new_constraints_for_children])

        while stack and stack[-1][1] == 0:
            stack.pop()

    valency = int(sum(s[1] for s in stack))
    current_forbidden_set = set(stack[-1][2]) if stack else set()
    if c_id is not None and pow_id is not None and stack:
        if stack[-1][0] == pow_id and stack[-1][1] == 1:
            current_forbidden_set.add(c_id)
    return valency, current_forbidden_set


def analyze_prefix_tree_context(
        self: "PrefixRepairHelper",
        seq: Sequence[int],
        transcendental_ids: Optional[Sequence[int]] = None,
        pow_id: Optional[int] = None,
        c_id: Optional[int] = None,
        start_id: Optional[int] = None,
) -> Tuple[int, set]:
    return _analyze_prefix_tree_context_impl(
        helper=self,
        seq=seq,
        transcendental_ids=transcendental_ids,
        pow_id=pow_id,
        c_id=c_id,
        start_id=start_id,
    )


PrefixRepairHelper.analyze_prefix_tree_context = analyze_prefix_tree_context


def constrained_decode_batch_from_position_logits(
        output_logits: torch.Tensor,
        helper: PrefixRepairHelper,
        seq_len: int,
        allowed_leaf_ids_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Decode full packed sequences from per-position logits with prefix-grammar constraints.

    Args:
        output_logits: AR logits. Accepts shape [L-1,B,V] or [B,L-1,V] where L == seq_len.
        seq_len: desired packed length [S] body [F] PAD...
        allowed_leaf_ids_batch: optional list of allowed leaf ids per sample.

    Returns:
        tokens: [B,seq_len] packed sequences, each prefix-valid.
    """
    if output_logits.dim() != 3:
        raise ValueError("output_logits must be rank-3")

    # Normalize to [L-1,B,V]
    if output_logits.shape[0] == seq_len - 1:
        logits = output_logits  # [L-1,B,V]
    elif output_logits.shape[1] == seq_len - 1:
        logits = output_logits.permute(1, 0, 2).contiguous()
    else:
        # best-effort: treat first dim as time
        logits = output_logits

    T, B, V = logits.shape
    max_body_len = max(1, seq_len - 2)
    T_use = min(T, max_body_len)

    out = torch.full((B, seq_len), int(helper.pad_id), dtype=torch.long, device=logits.device)
    out[:, 0] = int(helper.start_id)

    for b in range(B):
        leaf_ids = allowed_leaf_ids_batch[b] if (allowed_leaf_ids_batch is not None and b < len(
            allowed_leaf_ids_batch)) else helper.allowed_leaf_ids_from_body([])
        slot_logits = logits[:T_use, b, :]  # [T_use,V]
        body = constrained_decode_body_from_slot_logits(
            slot_logits=slot_logits,
            helper=helper,
            allowed_leaf_ids=leaf_ids,
            max_body_len=T_use,
        )
        if body is None or len(body) == 0 or (not helper.validate_body(body)):
            body = [int(helper._random_const_leaf())]

        packed = helper.pack(body, max_len=seq_len)
        out[b, :len(packed)] = torch.tensor(packed, dtype=torch.long, device=out.device)

    return out



def _select_argmax_from_allowed(logits_1d, allowed_ids):
    """Select argmax token id from a restricted candidate set.

    Args:
        logits_1d: 1D logits tensor [V] or array-like.
        allowed_ids: iterable of allowed token ids.

    Returns:
        int token id in allowed_ids with maximum logit.
    """
    if logits_1d is None:
        raise ValueError("logits_1d is None")
    # Convert candidates to a flat list[int]
    if isinstance(allowed_ids, (set, tuple)):
        allowed = list(allowed_ids)
    else:
        allowed = list(allowed_ids)
    if len(allowed) == 0:
        raise ValueError("allowed_ids is empty")

    # Torch fast path
    try:
        import torch
        if isinstance(logits_1d, torch.Tensor):
            cand = torch.as_tensor(allowed, dtype=torch.long, device=logits_1d.device)
            cand_logits = logits_1d.index_select(0, cand)
            return int(cand[int(torch.argmax(cand_logits).item())].item())
    except Exception:
        pass

    # Numpy fallback
    import numpy as np
    arr = np.asarray(logits_1d)
    best = allowed[0]
    best_v = float(arr[best])
    for tid in allowed[1:]:
        v = float(arr[tid])
        if v > best_v:
            best_v = v
            best = tid
    return int(best)


def constrained_decode_body_from_slot_logits(
        slot_logits: torch.Tensor,
        helper: PrefixRepairHelper,
        allowed_leaf_ids: Sequence[int],
        max_body_len: int,
) -> Optional[List[int]]:
    """Decode a prefix body from a sequence of slots (variable-length), each with logits over vocab.

    Key invariant: every prefix during decoding must remain *closable* within the remaining slots.
    That is, after selecting a token with arity a, we require:
        new_need = need - 1 + a
        new_need <= remaining_slots_after_this
    Because each future slot can reduce need by at most 1 (choosing a leaf).
    """
    if slot_logits.dim() != 2:
        raise ValueError("slot_logits must be [S,V]")
    S, V = slot_logits.shape
    max_body_len = int(max(1, max_body_len))
    S = min(S, max_body_len)

    # Filter candidate pools and exclude invalid tokens.
    specials = {int(helper.pad_id), int(helper.start_id), int(helper.finish_id)}
    leaf_pool = [int(t) for t in allowed_leaf_ids]
    leaf_pool = [t for t in leaf_pool if t not in specials and helper.arity(t) == 0]

    unary = [int(t) for t in helper.unary_ids]
    unary = [t for t in unary if t not in specials and helper.arity(t) == 1]
    binary = [int(t) for t in helper.binary_ids]
    binary = [t for t in binary if t not in specials and helper.arity(t) == 2]

    need = 1
    body: List[int] = []
    for i in range(S):
        remaining = S - i
        if need <= 0:
            break
        if remaining < need:
            return None

        rem_after = remaining - 1

        # Build allowed list with closure constraint
        allowed: List[int] = []
        if remaining == need:
            # must pick leaf
            for t in leaf_pool:
                if (need - 1) <= rem_after:
                    allowed.append(t)
        else:
            # leaf
            if (need - 1) <= rem_after:
                allowed.extend(leaf_pool)
            # unary
            if need <= rem_after:
                allowed.extend(unary)
            # binary
            if (need + 1) <= rem_after:
                allowed.extend(binary)

        if not allowed:
            return None

        tok = int(_select_argmax_from_allowed(slot_logits[i], allowed))
        a = int(helper.arity(tok))
        if a < 0:
            return None
        new_need = need - 1 + a

        # forced token must also obey closability
        if new_need > rem_after:
            return None

        body.append(tok)
        need = new_need
        if need == 0:
            break

    if not helper.validate_body(body):
        return None
    return body


def constrained_decode_batch_from_edit_logits(
        replace_logits: torch.Tensor,
        insert_logits: torch.Tensor,
        cur_tokens: torch.Tensor,
        op_tags: torch.Tensor,
        ins_counts: torch.Tensor,
        helper: PrefixRepairHelper,
        allowed_leaf_ids_batch: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Decode edited sequences using (op, ins_count) and generator logits.

    replace_logits: [B,L,V]
    insert_logits:  [B,L,K,V]
    cur_tokens:     [B,L]  (x_t)
    op_tags:        [B,L]  (0=KEEP,1=DELETE,2=REPLACE)
    ins_counts:     [B,L]  (0..K)
    Returns: [B,L] (x_{t-1}) packed and prefix-valid.
    """
    B, L, V = replace_logits.shape
    K = int(insert_logits.shape[2])
    max_body_len = max(1, L - 2)
    out = torch.full((B, L), int(helper.pad_id), dtype=torch.long, device=replace_logits.device)

    for b in range(B):
        allowed_leaf = allowed_leaf_ids_batch[b] if allowed_leaf_ids_batch is not None else helper.default_leaf_ids
        seq = cur_tokens[b].detach().cpu().tolist()
        body = helper.extract_body(seq)
        body_len = min(len(body), max_body_len)

        slots: List[torch.Tensor] = []
        for i in range(body_len):
            pos = 1 + i
            tok = int(cur_tokens[b, pos].item())
            tag = int(op_tags[b, pos].item())
            if tag == TAG_DELETE:
                continue

            if tag == TAG_KEEP:
                slots.append(_force_logits_to_token(replace_logits[b, pos], tok))
            else:
                # REPLACE
                slots.append(replace_logits[b, pos])

            m = int(ins_counts[b, pos].item())
            if m > 0:
                m = min(m, K)
                for j in range(m):
                    slots.append(insert_logits[b, pos, j])

        if not slots:
            # extreme fallback: a single const leaf
            body_new = [helper._random_const_leaf()]
        else:
            slot_logits = torch.stack(slots, dim=0)
            body_new = constrained_decode_body_from_slot_logits(slot_logits, helper, allowed_leaf, max_body_len)
            if body_new is None:
                # strict fallback: keep the current body (do not project/make-valid)
                body_new = body[:max_body_len] if body else [helper._random_const_leaf()]

        packed = helper.pack(body_new, max_len=L)
        out[b] = torch.tensor(packed, dtype=torch.long, device=replace_logits.device)

    return out
