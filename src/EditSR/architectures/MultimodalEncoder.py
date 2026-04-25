import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(dim_split), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class MultiModalEncoder(pl.LightningModule):
    """NeSymReS SetEncoder integrated into a single file for this project.

    The encoder consumes the raw point features ([x_1..x_n, y]) and internally
    performs the same float->bit preprocessing used by native NeSymReS when
    ``bit16=True``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.linear = bool(getattr(cfg, 'linear', False))
        self.bit16 = bool(getattr(cfg, 'bit16', True))
        self.norm = bool(getattr(cfg, 'norm', True))
        if self.linear == self.bit16:
            raise AssertionError('one and only one between linear and bit16 must be true at the same time')

        self.activation = str(getattr(cfg, 'activation', 'relu'))
        self.input_normalization = bool(getattr(cfg, 'input_normalization', False))

        self.dim_input = int(getattr(cfg, 'dim_input'))
        self.dim_hidden = int(getattr(cfg, 'dim_hidden'))
        self.num_heads = int(getattr(cfg, 'num_heads'))
        self.num_inds = int(getattr(cfg, 'num_inds', 50))
        self.num_features = int(getattr(cfg, 'num_features', getattr(cfg, 'num_queries', 10)))
        self.n_l_enc = int(getattr(cfg, 'n_l_enc', getattr(cfg, 'n_l_points_encoder', 4)))
        self.ln = bool(getattr(cfg, 'ln', True))

        if self.linear:
            self.linearl = nn.Linear(self.dim_input, 16 * self.dim_input)

        self.selfatt = nn.ModuleList()
        self.selfatt1 = ISAB(16 * self.dim_input, self.dim_hidden, self.num_heads, self.num_inds, ln=self.ln)
        for _ in range(self.n_l_enc):
            self.selfatt.append(ISAB(self.dim_hidden, self.dim_hidden, self.num_heads, self.num_inds, ln=self.ln))
        self.outatt = PMA(self.dim_hidden, self.num_heads, self.num_features, ln=self.ln)

    def float2bit(self, f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        s = (torch.sign(f + 0.001) * -1 + 1) * 0.5
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float('-inf')] = -(2 ** (num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2 ** (num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        f2 = f1 / 2 ** e_scientific
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.dtype
        exponent_bits = torch.arange(num_bits, device=remainder.device, dtype=dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.dtype
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device=integer.device, dtype=dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0], x.shape[1], -1)
            if self.norm:
                x = (x - 0.5) * 2

        if self.input_normalization and not self.bit16:
            means = x[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = x[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            x[:, :, -1] = (x[:, :, -1] - means) / std

        if self.linear:
            if self.activation == 'relu':
                x = torch.relu(self.linearl(x))
            elif self.activation == 'sine':
                x = torch.sin(self.linearl(x))
            else:
                x = self.linearl(x)

        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        return x
