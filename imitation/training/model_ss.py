#!/usr/bin/env python3
"""The marnie-sixthsense set-transformer: policy over legal options + WDL value.

Implements docs/architecture.pdf (model sections): pre-RMSNorm, SwiGLU, QK-Norm, a relational
attention bias over typed edges, Set-Transformer PMA pooling, and a policy head
that scores option tokens directly.

Illegal actions are never constructed, so they have no logit — legality is a
property of the input, not a mask on the output. Padding is masked.

    python training/model_ss.py --self-test     # param count + overfit one batch
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch
import torch.nn.functional as F
from torch import nn

# card_tok() emits CARD_BASE(2) + cardId, and cardId runs 1..1267, so the
# largest index is 1269 and the table needs 1270 rows. 1267+2 was an
# off-by-one that only surfaces as a CUDA device-side assert.
CARD_VOCAB = 1267 + 3
ATTACK_VOCAB = 1556 + 3
N_FAMILY = 21
N_OWNER = 3
# Card semantics as categorical tokens (featurize.py N_CARD_TYPE/N_ENERGY_TYPE).
# Index 0 means 'this token carries no card'. 8 and 13 rows, every row common,
# so unlike the 1270-row card table (1040 rows never trained) these generalise
# to cards the corpus never contained.
N_CARD_TYPE = 8
N_ENERGY_TYPE = 13
# Card RULES TEXT as a shared concept basis. artifacts/card_text_bow.npz holds a
# FIXED (CARD_VOCAB x V) row-normalised bag of unigrams+bigrams over Skill.text
# and Attack.text, and the model learns a d-vector per concept:
#
#     card_text = card_bow @ word_emb      -> (CARD_VOCAB, d), once per forward
#     x        += card_text[card_id]       -> gather, free
#
# WHY THIS AND NOT MORE CARD-ID CAPACITY. The card-id table has 1270 rows and only
# 230 are ever trained; the other 1040 keep near-init magnitude (L2 11.38 vs 11.60)
# so an unseen card is full-scale noise. The concept basis has ~1651 rows at ~33
# occurrences each and every card decomposes onto it, so a card the corpus never
# contained still arrives as a mixture of concepts seen thousands of times.
# Fixed structure, learned meaning: card_bow gets no gradient, word_emb does.
#
# Bigrams are included because pure unigrams conflate "your Active" with "your
# opponent's Active" -- measured 'your opponent' 699 vs 'your active' 28.
N_OPTION_TYPE = 17
N_WDL = 2           # win / not-win; see the value head
NUM_W = 269         # must equal featurize.NUMERIC_WIDTH
OPT_NUM_W = 269     # must equal featurize.OPTION_NUMERIC_WIDTH
N_RELATIONS = 8   # 0 none, 1 same-owner, 2 same-zone, 3 ptr, 4 dest, 5 active-active, 6 prompt
# Per-column statistics are only valid in the scale space they were measured in,
# so a v2 checkpoint stamps which space that was and a loader can refuse a
# mismatch instead of silently crushing every column onto one value.
_SCALE_IDS = {"layernorm": 0, "log1p": 1}

def _scale_id(name):
    """Map a scale name to its stamped id, REFUSING an unknown one.

    Was . A typo -- "log1P", "lognorm" -- became
    -1, which the loader could only read as "unstamped", i.e. indistinguishable
    from an old checkpoint that predates the stamp. The scale is the difference
    between log1p and layernorm numerics; guessing it is the I-1 shape.
    """
    if name not in _SCALE_IDS:
        raise ValueError(
            f"unknown num_scale {name!r}; expected one of {sorted(_SCALE_IDS)}. "
            f"An unrecognised scale cannot be stamped, and an unstamped scale is "
            f"served by guessing.")
    return _SCALE_IDS[name]



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    """Bidirectional over the token set. QK-Norm is not optional at this size:
    without it attention logits grow unbounded, the softmax saturates and the
    loss plateaus in a way that looks like the task being too hard."""

    def __init__(self, d: int, heads: int, layer: int, attn_dropout: float = 0.0):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.dh)
        self.k_norm = RMSNorm(self.dh)
        self.rel_bias = nn.Parameter(torch.zeros(N_RELATIONS, heads))
        # Dropout on the attention probabilities, i.e. AFTER the softmax. SDPA
        # applies it internally, which keeps the fused kernel; doing it by hand
        # would materialise the (b, heads, n, n) matrix this class exists to avoid.
        self.attn_dropout = float(attn_dropout)

    def forward(self, x, mask, rel):
        # Fused SDPA rather than a materialised (b, heads, n, n) score matrix. The
        # hand-rolled form was memory-bound: throughput FELL from batch 128 to 4096.
        # Same maths (QK-Norm and key padding preserved) - measured max abs diff on
        # policy logits 1.49e-07 - at 1.3-1.4x, or 2.2-3.9x under torch.compile.
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q.view(b, n, self.h, self.dh)).transpose(1, 2)
        k = self.k_norm(k.view(b, n, self.h, self.dh)).transpose(1, 2)
        v = v.view(b, n, self.h, self.dh).transpose(1, 2)
        key_pad = mask[:, None, None, :]
        if rel is not None:
            # rel: (b, n, n) relation ids -> (b, n, n, heads) -> (b, heads, n, n)
            # F.embedding, NOT self.rel_bias[rel]. Bitwise-identical forward,
            # completely different backward. Tensor.__getitem__ backprops through
            # index_put_(accumulate=True): b*n*n = 4.8M atomicAdds onto
            # N_RELATIONS*heads = 64 float addresses, per block, per step, fully
            # serialised. MEASURED at batch 96, 224 tokens, fp32, one RTX PRO 6000:
            #   rel=None                461 ms/batch
            #   self.rel_bias[rel]   40,719 ms/batch    88x
            #   F.embedding             735 ms/batch   1.59x
            bias = F.embedding(rel, self.rel_bias).permute(0, 3, 1, 2)
            attn_mask = bias.masked_fill(~key_pad, torch.finfo(bias.dtype).min)
        else:
            attn_mask = key_pad
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=1.0 / math.sqrt(self.dh),
            # must be 0 outside training, or eval and the exported NumPy agent
            # stop agreeing and the export-parity gate fails
            dropout_p=self.attn_dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        # on the hidden activation, before the down projection -- the standard
        # placement, and the widest tensor in the block (hidden > d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, d, heads, hidden, layer, attn_dropout=0.0,
                 resid_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attention(d, heads, layer, attn_dropout=attn_dropout)
        self.ffn = SwiGLU(d, hidden, dropout=ffn_dropout)
        # Residual dropout: applied to each sublayer's OUTPUT before it is added
        # back, so the skip path stays clean. Dropping the residual stream itself
        # would delete the identity path this depth relies on.
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x, mask, rel):
        x = x + self.drop(self.attn(self.n1(x), mask, rel))
        return x + self.drop(self.ffn(self.n2(x)))


class PMA(nn.Module):
    """Set-Transformer pooling by a learned seed, not a mean a wide bench dilutes."""

    def __init__(self, d, heads, seeds: int = 1):
        super().__init__()
        # seeds > 1 gives each consumer group its own pooled read of the state.
        # With one seed the value head, all aux heads and the policy's state half
        # share a single d-dim vector; measured effect share splits along exactly
        # that routing boundary (pool-routed families <=1.8%, direct-routed 69.1%
        # and 23.6%). Default 1 keeps v1 byte-identical: same shape, same draw.
        self.seeds = int(seeds)
        self.seed = nn.Parameter(torch.randn(1, self.seeds, d) * 0.02)
        self.h, self.dh = heads, d // heads
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, mask):
        b, n, d = x.shape
        s = self.seeds
        q = self.seed.expand(b, s, d).view(b, s, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(b, n, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(b, n, self.h, self.dh).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        logits = logits.masked_fill(~mask[:, None, None, :], torch.finfo(logits.dtype).min)
        y = (logits.softmax(-1) @ v).transpose(1, 2)
        # seeds == 1 reshapes to (b, d), the exact shape v1 fed self.out. Keeping
        # the original path is free, so it is kept; but note that the 3-D form
        # (reshape to (b, 1, d), Linear, squeeze) was MEASURED bitwise identical
        # to this one on CPU fp32 under torch 2.13, with and without bias, at
        # n = 1/3/8/64. So this line is literal preservation, NOT a fix for an
        # observed discrepancy -- do not cite it as one. Untested on CUDA.
        y = y.reshape(b, d) if s == 1 else y.reshape(b, s, d)
        return self.out(y)


class SixthSenseNet(nn.Module):
    def __init__(self, d=384, layers=12, heads=12, hidden=1024, n_aux=6,
                 dropout=0.15, num_scale="layernorm", aux_classes=None,
                 block_dropout=0.0, card_id_dropout=0.0, card_bow=None,
                 num_w=None, opt_num_w=None, n_wdl=None, card_semantics=True,
                 arch_v2=False, pool_seeds=4):
        # num_w / opt_num_w / n_wdl / card_semantics default to the module
        # globals, so training reproduces exactly. They exist so the ARENA can
        # rebuild an OLD checkpoint (64-wide numerics, 3-class value head, no
        # card_type/energy_type) inside the same process as a new one -- without
        # them SixthSenseNet always builds today's shape and load_state_dict
        # rejects every historical checkpoint.
        # `dropout` has only ever reached the embedding (one nn.Dropout, applied
        # once before the blocks), so every arm run to date regularised ~5% of the
        # parameters and left the 12 blocks bare. That is why dropout 0.15 vs 0.0
        # measured as noise (52.28% over 1600 paired games) -- it was barely an
        # intervention. `block_dropout` drives the three standard placements:
        # attention probabilities, FFN hidden activation, and sublayer residuals.
        # Default 0.0 so existing runs reproduce exactly.
        super().__init__()
        num_w = NUM_W if num_w is None else int(num_w)
        opt_num_w = OPT_NUM_W if opt_num_w is None else int(opt_num_w)
        n_wdl = N_WDL if n_wdl is None else int(n_wdl)
        self.card_semantics = bool(card_semantics)
        self.card = nn.Embedding(CARD_VOCAB, d, padding_idx=0)
        self.family = nn.Embedding(N_FAMILY, d)
        self.owner = nn.Embedding(N_OWNER, d)
        if self.card_semantics:
            self.card_type = nn.Embedding(N_CARD_TYPE, d, padding_idx=0)
            self.energy_type = nn.Embedding(N_ENERGY_TYPE, d, padding_idx=0)
        else:
            self.card_type = self.energy_type = None
        # card_bow is a BUFFER: it ships inside the state_dict so the exported
        # NumPy agent gets it for free, and it never receives a gradient.
        if card_bow is not None:
            bow = torch.as_tensor(card_bow, dtype=torch.float32)
            if bow.shape[0] != CARD_VOCAB:
                raise ValueError(
                    f"card_bow has {bow.shape[0]} rows, expected CARD_VOCAB="
                    f"{CARD_VOCAB}; rebuild the card-text basis from your own engine install")
            self.register_buffer("card_bow", bow)
            self.word = nn.Embedding(bow.shape[1], d)
        else:
            self.card_bow, self.word = None, None
        self.num_scale = num_scale
        self.num_norm = nn.LayerNorm(num_w)
        self.num_proj = nn.Linear(num_w, d)

        # ---------------------------------------------------------------- v2
        # OFF BY DEFAULT. Every v2 tensor is created only when arch_v2 is True,
        # so a v1 state_dict has exactly the keys it always had and the v1
        # forward output is bit-identical to the pre-v2 file. Both are asserted
        # by scripts/verify_v2.py, the second against a pristine module copy.
        self.arch_v2 = bool(arch_v2)
        if self.arch_v2 and int(pool_seeds) < 3:
            # Under v2 the forward reads seed 0 (policy), 1 (value) and 2 (aux).
            # With 2 seeds the aux read aliased onto the POLICY read and with 1 it
            # raised IndexError after the data had already been loaded. Both are
            # plausible ablations, so they fail here at construction rather than
            # mid-epoch or, worse, silently.
            raise ValueError(
                f"arch_v2 needs pool_seeds >= 3 (policy/value/aux); got {pool_seeds}")
        if self.arch_v2:
            # (1) FAMILY-CONDITIONED NUMERIC PROJECTION.
            # One shared nn.Linear is applied to every family, but column meaning
            # is family-dependent: column 8 is bench_slot_0 on a BENCH token and
            # HAND POSITION 0 on a HAND token (measured attribution 0.0241 from
            # HAND vs 0.0003 from BENCH). Family enters v1 only ADDITIVELY, and an
            # additive term cannot modulate a linear map.
            # FiLM rather than a per-family weight matrix: N_FAMILY*num_w*2 params
            # against N_FAMILY*num_w*d for full per-family matrices. The best model
            # this project ever produced is 23.6M params and the current 45.4M has
            # never beaten it, so cheap conditioning is the justified form.
            # gamma init 1, beta init 0 => identity at step 0.
            self.fam_gamma = nn.Embedding(N_FAMILY, num_w)
            self.fam_beta = nn.Embedding(N_FAMILY, num_w)
            nn.init.ones_(self.fam_gamma.weight)
            nn.init.zeros_(self.fam_beta.weight)
            # (3) PER-COLUMN STANDARDIZATION, in the space the projection reads.
            # Measured raw spread 39,570x; log1p leaves 891x; learned column norms
            # span only 9.27x, so the model does NOT compensate. 5 of 160 columns
            # carry 50% of input variance; 52 binary flags carry 2.3%.
            # Buffers, not parameters: they are corpus statistics, and they ship
            # in the state_dict so serving cannot disagree with training.
            self.register_buffer("num_mu", torch.zeros(num_w))
            self.register_buffer("num_sigma", torch.ones(num_w))
            # The option half gets the same treatment. The spread argument is a
            # property of the columns, not of which side of the concat they sit
            # on, and OPTION carries the largest measured share of decision
            # effect. FiLM is correctly absent here -- option tokens carry no
            # family index, so there is nothing to condition on.
            self.register_buffer("opt_num_mu", torch.zeros(opt_num_w))
            self.register_buffer("opt_num_sigma", torch.ones(opt_num_w))
            # _standardize runs AFTER _scale, so the stats are only meaningful in
            # that space. Stamped so a loader can refuse a mismatch.
            self.register_buffer(
                "num_scale_id",
                torch.tensor(_scale_id(num_scale), dtype=torch.long))
        else:
            self.fam_gamma = self.fam_beta = None
        self.attack = nn.Embedding(ATTACK_VOCAB, 96, padding_idx=0)
        self.attack_proj = nn.Linear(96, d, bias=False)
        self.opt_type = nn.Embedding(N_OPTION_TYPE + 1, d)
        self.opt_num_norm = nn.LayerNorm(opt_num_w)
        self.opt_num = nn.Linear(opt_num_w, d)
        self.ptr_src = nn.Linear(d, d, bias=False)
        self.ptr_dst = nn.Linear(d, d, bias=False)
        self.emb_norm = RMSNorm(d)
        self.drop = nn.Dropout(dropout)
        # CARD-ID DROPOUT. Only 230 of 1270 card rows are ever looked up (18.1%),
        # and the untrained 1040 keep near-init magnitude -- measured L2 11.38 vs
        # 11.60 for trained rows, a ratio of 0.981. So an opponent playing a card
        # we never saw feeds attention a full-scale random vector.
        #
        # The information is not actually missing: every card token also carries
        # 64 numeric features (hp, maxHp, damage, damage ratio, weakness and
        # resistance against the ACTUAL defender, 21 effect tags). Those are
        # computed from real card properties and are correct for any card.
        #
        # Randomly zeroing the card id during training (index 0 is padding_idx,
        # so it embeds to the zero vector) forces the model to read those features
        # instead of memorising 230 identities. Same target as a JEPA masked-card
        # objective, without the extra head.
        self.card_id_dropout = float(card_id_dropout)
        self.blocks = nn.ModuleList(
            [Block(d, heads, hidden, i, attn_dropout=block_dropout,
                   resid_dropout=block_dropout, ffn_dropout=block_dropout)
             for i in range(layers)])
        self.final = RMSNorm(d)
        # (2) POOLING. v1 uses ONE seed, and that single vector is the only input
        # to the value head, every aux head, and the policy's state half -- all of
        # them through one d-dim bottleneck. Measured family share of decision
        # effect splits cleanly along the routing boundary: OPTION 69.1% and HAND
        # 23.6% (both reach the policy WITHOUT the pool) against ACTIVE 1.5%,
        # BENCH 1.8%, GLOBAL/PLAYER <0.5% (all pool-routed). Under search the
        # value head becomes the frontier evaluator, so starving it is
        # disqualifying rather than merely suboptimal.
        # v2 gives each consumer group its own seed: 0 policy, 1 value, 2+ aux.
        self.pool = PMA(d, heads, seeds=pool_seeds) if arch_v2 else PMA(d, heads)
        self.policy = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, 1))
        # 2 classes: win / not-win. Draws are folded into losses by the dataset
        # builder (measured draw base rate 0.000) because the objective is to WIN,
        # not to avoid losing -- a line that secures a draw is a line that failed.
        self.value = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, n_wdl))
        # Per-head class counts. Every head used to emit 4 logits regardless of
        # its target, which only worked because the trainer fed all of them the
        # 3-class game outcome. next_action_type alone needs 18.
        self.aux_classes = list(aux_classes) if aux_classes else [4] * n_aux
        self.aux = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d // 2), nn.SiLU(), nn.Linear(d // 2, c))
             for c in self.aux_classes])

    # Anything smaller than this is a constant column that measurement noise gave
    # a nonzero sigma. Dividing by it manufactures an enormous activation from a
    # column carrying no information: sigma=1e-9 turned an input of 1.0 into 1e9.
    SIGMA_FLOOR = 1e-6

    @staticmethod
    def _check_stats(mu, sigma, buf_mu, buf_sigma, what):
        """Reject the stats files that would otherwise install themselves quietly.

        Every rejection here is a measured failure, not a hypothetical: copy_
        BROADCASTS, so a file holding one global mean/std rather than per-column
        arrays was accepted and fanned out across every column; and
        `live = sigma > 0` is False for negative AND for NaN, so a corrupt file
        turned standardization off on every column while the run reported success.
        """
        mu = torch.as_tensor(mu, dtype=buf_mu.dtype).reshape(-1)
        sigma = torch.as_tensor(sigma, dtype=buf_sigma.dtype).reshape(-1)
        for name, t, buf in (("mu", mu, buf_mu), ("sigma", sigma, buf_sigma)):
            if t.shape != buf.shape:
                raise ValueError(
                    f"{what} {name} has width {tuple(t.shape)}, model expects "
                    f"{tuple(buf.shape)} -- a scalar or length-1 array would "
                    f"otherwise broadcast silently")
        if not torch.isfinite(mu).all():
            raise ValueError(f"{what} mu contains non-finite values")
        if not torch.isfinite(sigma).all():
            raise ValueError(f"{what} sigma contains non-finite values "
                             f"(NaN reads as 'not live' and disables the column)")
        if (sigma < 0).any():
            raise ValueError(f"{what} sigma contains negative values "
                             f"(negative reads as 'not live' and disables the column)")
        sigma = torch.where(sigma < SixthSenseNet.SIGMA_FLOOR,
                            torch.zeros_like(sigma), sigma)
        return mu, sigma

    def set_num_stats(self, mu, sigma, opt_mu=None, opt_sigma=None):
        """Install per-column statistics measured on the corpus they will be used
        with. sigma == 0 marks a constant column and is passed through untouched."""
        if not self.arch_v2:
            raise RuntimeError("set_num_stats requires arch_v2=True")
        mu, sigma = self._check_stats(mu, sigma, self.num_mu, self.num_sigma, "state")
        with torch.no_grad():
            self.num_mu.copy_(mu)
            self.num_sigma.copy_(sigma)
            if opt_mu is not None and opt_sigma is not None:
                o_mu, o_sigma = self._check_stats(
                    opt_mu, opt_sigma, self.opt_num_mu, self.opt_num_sigma, "option")
                self.opt_num_mu.copy_(o_mu)
                self.opt_num_sigma.copy_(o_sigma)

    @staticmethod
    def _apply_standardize(z, mu, sigma):
        live = sigma > 0
        denom = torch.where(live, sigma, torch.ones_like(sigma))
        mu = torch.where(live, mu, torch.zeros_like(mu))
        return (z - mu) / denom

    def _standardize(self, z):
        """(z - mu) / sigma, with sigma == 0 PASSING THROUGH rather than dividing.
        127 of 320 columns measure exactly constant here, so a naive divide would
        produce inf/nan on more than a third of the width."""
        if not self.arch_v2:
            return z
        return self._apply_standardize(z, self.num_mu, self.num_sigma)

    def _standardize_opt(self, z):
        if not self.arch_v2:
            return z
        return self._apply_standardize(z, self.opt_num_mu, self.opt_num_sigma)

    def _num_project(self, z, fam=None):
        """v1: one shared linear for every family. v2: FiLM-conditioned on family
        first, so the same numeric vector under two families projects differently."""
        if self.arch_v2 and fam is not None:
            z = z * self.fam_gamma(fam) + self.fam_beta(fam)
        return self.num_proj(z)

    def _scale(self, x):
        if self.num_scale == "log1p":
            return torch.sign(x) * torch.log1p(x.abs())
        return self.num_norm(x)

    def _scale_opt(self, x):
        if self.num_scale == "log1p":
            return torch.sign(x) * torch.log1p(x.abs())
        return self.opt_num_norm(x)

    def forward(self, batch):
        fam, own, card, num = batch["family"], batch["owner"], batch["card"], batch["numeric"]
        mask = batch["mask"]
        if self.training and self.card_id_dropout > 0.0:
            # index 0 is padding_idx -> the zero vector, i.e. "identity withheld".
            # Training only: eval must stay deterministic or export parity fails.
            keep = torch.rand(card.shape, device=card.device) >= self.card_id_dropout
            card = card * keep
        x = (self.family(fam) + self.owner(own) + self.card(card)
             + self._num_project(self._standardize(self._scale(num)),
                                 fam if self.arch_v2 else None))
        # Optional so older checkpoints and callers keep working unchanged.
        ct, et = batch.get('card_type'), batch.get('energy_type')
        if ct is not None and self.card_type is not None:
            x = x + self.card_type(ct)
        if et is not None and self.energy_type is not None:
            x = x + self.energy_type(et)
        if self.word is not None:
            # (V_cards x V_concepts) @ (V_concepts x d) once, then gather by id.
            # Embedding K words per TOKEN would materialise (b, n, K, d) ~ 600M
            # floats a batch; this is one small matmul and a lookup.
            x = x + (self.card_bow @ self.word.weight)[card]
        n_state = x.shape[1]
        opt = (self.opt_type(batch["option_type"])
               + self.opt_num(self._standardize_opt(
                   self._scale_opt(batch["option_numeric"]))))
        atk = batch.get("option_attack")
        if atk is not None:
            # two attacks with equal cost and damage were previously identical
            opt = opt + self.attack_proj(self.attack(atk))
        # Pointer copy: add the embedding of the entity each option refers to,
        # and of its destination for ATTACH/EVOLVE. -1 means "points at nothing"
        # (END, RETREAT, YES/NO) and contributes zero.
        ptr = batch.get("option_ptr")
        if ptr is not None:
            safe = ptr.clamp(min=0)
            b_idx = torch.arange(x.shape[0], device=x.device)[:, None, None]
            gathered = x[b_idx, safe]                    # (b, no, 2, d)
            gathered = gathered * (ptr >= 0).unsqueeze(-1).to(gathered.dtype)
            opt = opt + self.ptr_src(gathered[:, :, 0]) + self.ptr_dst(gathered[:, :, 1])
        x = torch.cat([x, opt], dim=1)
        x = self.drop(self.emb_norm(x))
        rel = batch.get("relation")
        for block in self.blocks:
            x = block(x, mask, rel)
        x = self.final(x)
        pooled = self.pool(x, mask)
        if self.arch_v2:
            # seed 0 -> policy, seed 1 -> value, seed 2.. -> aux (the spare seeds
            # are cycled if there are more aux heads than seeds). Each consumer
            # gets its own read of the board instead of all of them sharing one.
            # min()/cycle rather than a bare modulo over ALL seeds: with 2 seeds
            # `2 % 2` aliased aux onto seed 0, the POLICY read -- the worst
            # available pairing and the exact thing multi-seed pooling exists to
            # prevent. pool_seeds >= 3 is enforced in __init__, so the clamps here
            # are belt-and-braces rather than the guard.
            last = pooled.shape[1] - 1
            s_pol = pooled[:, 0, :]
            s_val = pooled[:, min(1, last), :]
            aux_in = [pooled[:, 2 + i % max(1, last - 1), :]
                      for i in range(len(self.aux))]
        else:
            s_pol = s_val = pooled
            aux_in = [pooled] * len(self.aux)
        opt_tok = x[:, n_state:, :]
        joined = torch.cat([opt_tok, s_pol[:, None, :].expand_as(opt_tok)], dim=-1)
        logits = self.policy(joined).squeeze(-1)
        # An option that does not exist has no logit at all.
        logits = logits.masked_fill(~batch["option_mask"], torch.finfo(logits.dtype).min)
        return {"policy": logits, "value": self.value(s_val),
                "aux": [head(z) for head, z in zip(self.aux, aux_in)]}


def param_report(model: nn.Module) -> dict:
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
    total = sum(groups.values())
    deployed = total - sum(v for k, v in groups.items() if k == "aux")
    return {"total": total, "deployed": deployed,
            "mb_fp16_deployed": round(deployed * 2 / 1e6, 2), "groups": groups}


def self_test(d, layers, heads, hidden, steps, device):
    torch.manual_seed(0)
    model = SixthSenseNet(d=d, layers=layers, heads=heads, hidden=hidden).to(device)
    rep = param_report(model)
    b, n_state, n_opt = 8, 48, 16
    n = n_state + n_opt
    batch = {
        "family": torch.randint(1, N_FAMILY, (b, n_state), device=device),
        "owner": torch.randint(0, N_OWNER, (b, n_state), device=device),
        "card": torch.randint(0, CARD_VOCAB, (b, n_state), device=device),
        "numeric": torch.randn(b, n_state, NUM_W, device=device),
        "option_type": torch.randint(0, N_OPTION_TYPE, (b, n_opt), device=device),
        "option_numeric": torch.randn(b, n_opt, OPT_NUM_W, device=device),
        "mask": torch.ones(b, n, dtype=torch.bool, device=device),
        "option_mask": torch.ones(b, n_opt, dtype=torch.bool, device=device),
    }
    target = torch.randint(0, n_opt, (b,), device=device)
    value_target = torch.randint(0, N_WDL, (b,), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
    first = last = None
    for i in range(steps):
        out = model(batch)
        loss = F.cross_entropy(out["policy"], target) + \
            F.cross_entropy(out["value"], value_target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i == 0:
            first = loss.item()
        last = loss.item()
    with torch.no_grad():
        acc = (model(batch)["policy"].argmax(-1) == target).float().mean().item()
    return rep, first, last, acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", type=str)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep, first, last, acc = self_test(args.d, args.layers, args.heads,
                                      args.hidden, args.steps, device)
    spec_total = 23.4e6
    within = abs(rep["total"] - spec_total) / spec_total
    status = "PASS" if within <= 0.10 and acc >= 0.99 and last < first * 0.1 else "FAIL"
    report = {
        "schema": "ptcg-ai/model-gate/v1", "status": status, "device": device,
        "config": {"d": args.d, "layers": args.layers, "heads": args.heads,
                   "hidden": args.hidden},
        "params": rep, "spec_total": spec_total,
        "within_spec_frac": round(within, 4),
        "overfit": {"loss_first": first, "loss_last": last, "argmax_accuracy": acc,
                    "steps": args.steps},
    }
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    sys.exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
