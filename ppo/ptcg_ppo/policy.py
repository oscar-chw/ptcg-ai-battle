"""The cardinality head, grafted onto the champion without disturbing it.

WHAT GETS ADDED
---------------
``SixthSenseNet`` scores each option from ``concat(option_token, pooled_state)``
through ``policy = Sequential(Linear(2d, d), SiLU, Linear(d, 1))``. STOP is scored
by the **same MLP** from ``concat(stop_token, pooled_state)``, so it is a genuine
per-board prediction rather than a constant, and it shares every weight the
policy head already learned::

    stop_token   (d,)                  one learned vector          d params
    stop_bias    (MAX_STOP_BIAS + 1,)  a learned offset per k      9 params

That is 521 new parameters on a 45,435,348-parameter model. Every existing tensor
loads unchanged, which is what makes the frozen parent an exact comparator rather
than an approximate one.

WHY ``stop_bias`` EXISTS SEPARATELY FROM ``stop_token``
--------------------------------------------------------
With a single static stop logit, ``P(stop)`` still rises on its own as options are
consumed -- the softmax denominator shrinks. That expresses "take everything above
a threshold", which is a good default. It cannot express "take exactly two,
whatever is on offer". ``stop_bias[k]`` adds that, at a cost of nine scalars, and
``--no-stop-bias`` ablates it. The measured pilot behaviour needs both shapes: at
``(0,2)`` and ``(0,5)`` the pilots take the cap, at ``(1,3)``/``(1,4)``/``(1,5)``
they take exactly one, and the project's own note is that no single constant can
express that inversion.

CALIBRATION, NOT A MAGIC NUMBER
--------------------------------
Gate G-TIE requires the extended policy to reproduce the champion's greedy moves
exactly at step 0. That holds iff the stop logit sits below the lowest option
logit that greedy would otherwise take, at every sub-step before ``maxCount``.
Rather than guess a bias and hope, ``calibrate_stop_bias`` measures the actual
gap on real board states and sets the bias to clear the observed worst case by a
margin -- then G-TIE verifies it on a held-out sample, because a calibration
checked on its own fit is not a check.

The bias must stay above the saturation floor: a STOP that can never be sampled
can never be credited, and the head would be decorative while every metric looked
fine. ``objective.check_stop_init`` refuses that case.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .action import MAX_STOP_BIAS
from .objective import GateRefusal, check_stop_init

# How far below the worst observed option logit to place the stop logit. Large
# enough that a board unlike the calibration sample still ties; small enough to
# stay well clear of the saturation floor.
TIE_MARGIN = 4.0


@dataclass(frozen=True)
class Architecture:
    """Derived from the checkpoint's tensors. Never from defaults, never from memory.

    ``train_ss.py`` defaults to ``--heads 12 --hidden 1024`` and will happily build
    a DIFFERENT architecture that still loads. The shapes are the only source of
    truth, and reading them costs one file open.
    """

    d_model: int
    n_layers: int
    n_heads: int
    numeric_width: int
    card_vocab: int
    n_relations: int
    ffn_hidden: int = 0
    n_wdl: int = 0
    aux_classes: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def architecture_from_state_dict(state: dict[str, torch.Tensor]) -> Architecture:
    """Read the architecture off the weights.

    Verified against FIXED-312, whose tensors give d512 / 12L / 8 heads /
    numeric_width 160 / 1270 cards -- and note the branch README describes a
    d384 12-head model, which is a DIFFERENT arm. Deriving beats believing.
    """
    missing = [k for k in ("num_proj.weight", "card.weight") if k not in state]
    if missing:
        raise GateRefusal(f"checkpoint is missing {missing}; not a SixthSenseNet")

    d_model, numeric_width = state["num_proj.weight"].shape
    card_vocab = state["card.weight"].shape[0]
    layers = {int(k.split(".")[1]) for k in state if k.startswith("blocks.")}
    if not layers:
        raise GateRefusal("checkpoint has no transformer blocks")
    n_layers = max(layers) + 1

    knorm = state.get("blocks.0.attn.k_norm.weight")
    if knorm is None:
        raise GateRefusal("cannot derive head count: blocks.0.attn.k_norm absent")
    head_dim = int(knorm.shape[0])
    if d_model % head_dim:
        raise GateRefusal(f"d_model {d_model} is not divisible by head_dim {head_dim}")
    n_heads = d_model // head_dim

    rel = state.get("blocks.0.attn.rel_bias")
    n_relations = int(rel.shape[0]) if rel is not None else 0

    # FFN width. FIXED-312 is 1365 and train_ss.py defaults to 1024 -- a model
    # built on the default would be a DIFFERENT network that loads most tensors
    # happily and only fails on the ones it happens to disagree about.
    ffn_hidden = 0
    for name in ("blocks.0.ffn.up.weight", "blocks.0.ffn.gate.weight",
                 "blocks.0.ff.0.weight"):
        if name in state:
            ffn_hidden = int(state[name].shape[0])
            break

    # Head widths, likewise derived. The value head is 2-class here (win /
    # not-win) where the constructor's default is 4, and the aux heads have
    # PER-HEAD class counts -- they are not uniform.
    n_wdl = int(state["value.2.weight"].shape[0]) if "value.2.weight" in state else 0
    aux_indices = sorted({int(k.split(".")[1]) for k in state
                          if k.startswith("aux.") and k.endswith(".2.weight")})
    aux_classes = tuple(int(state[f"aux.{i}.2.weight"].shape[0]) for i in aux_indices)

    return Architecture(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        numeric_width=numeric_width, card_vocab=card_vocab,
                        n_relations=n_relations, ffn_hidden=ffn_hidden,
                        n_wdl=n_wdl, aux_classes=aux_classes)


def assert_featurizer_width(arch: Architecture, featurizer_width: int) -> None:
    """GATE G-WIDTH. Refuse before a single GPU-second is spent.

    FIXED-312 is 160-wide; the workspace's current ``tools/featurize.py`` emits
    269. Feeding 269 columns into a ``(512, 160)`` projection is the defect class
    that once scored 355.7 with every offline gate green.
    """
    if featurizer_width != arch.numeric_width:
        raise GateRefusal(
            f"G-WIDTH FAILED: featurizer emits {featurizer_width} numeric columns "
            f"but num_proj expects {arch.numeric_width}. These must be the same "
            "featurizer, not two that agree in spirit. FIXED-312 is 160-wide; the "
            "workspace default is 269-wide. Use the 160-wide featurizer preserved "
            "at .gate/feat160/featurize.py, and ship that file itself rather than "
            "a copy -- a copy is what drifts."
        )


def build_model_from_state_dict(model_ss, state: dict[str, torch.Tensor],
                                arch: Architecture | None = None):
    """Construct the exact network the checkpoint describes, and load it strictly.

    Every constructor argument is DERIVED from the tensors. That is not
    fastidiousness: ``SixthSenseNet`` defaults to ``d=384, heads=12, hidden=1024,
    n_aux=6`` and ``model_ss.py`` additionally hardcodes ``NUM_W = 269`` under a
    comment reading "must equal featurize.NUMERIC_WIDTH". The champion is
    d512 / 8 heads / FFN 1365 / 160-wide, so every one of those defaults is wrong
    for it, and a model built on them loads most tensors and disagrees about the
    rest -- or, worse, agrees on shapes it should not.

    ``card_bow`` is a BUFFER carried inside the state dict, and ``self.word`` only
    exists when it is passed. Omitting it silently drops the card-semantics path
    that the champion was trained with.

    Raises on ANY missing or unexpected key. A partial load here is the
    substitution failure: an absent tensor becomes a plausible random one, and
    every downstream gate passes.
    """
    arch = arch or architecture_from_state_dict(state)
    kwargs: dict[str, object] = {
        "d": arch.d_model,
        "layers": arch.n_layers,
        "heads": arch.n_heads,
        "hidden": arch.ffn_hidden,
    }
    accepted = set(model_ss.SixthSenseNet.__init__.__code__.co_varnames)
    optional = {
        "num_w": arch.numeric_width,
        "opt_num_w": arch.numeric_width,
        "n_wdl": arch.n_wdl,
        "n_aux": len(arch.aux_classes),
        "aux_classes": list(arch.aux_classes),
    }
    for name, value in optional.items():
        if name in accepted and value:
            kwargs[name] = value
    if "card_bow" in accepted and "card_bow" in state:
        kwargs["card_bow"] = state["card_bow"]

    model = model_ss.SixthSenseNet(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise GateRefusal(
            "the constructed model is not the one the checkpoint describes.\n"
            f"  missing (model wants, checkpoint lacks):    {sorted(missing)[:10]}\n"
            f"  unexpected (checkpoint has, model lacks):   {sorted(unexpected)[:10]}\n"
            f"  constructed with: {kwargs}\n"
            "Do not load this partially. A missing tensor becomes a random one and "
            "every downstream gate goes green over it."
        )
    return model, kwargs


class StopHead(nn.Module):
    """Scores STOP through the host model's own policy MLP.

    Deliberately does not own a second MLP. A separate head would start
    uncorrelated with the option scores it has to be compared against, so its
    logit would be on a different scale and the comparison meaningless until it
    trained -- during which the agent would pass at random.
    """

    def __init__(self, d_model: int, policy_mlp: nn.Module,
                 stop_init_logit: float = -TIE_MARGIN,
                 use_bias: bool = True) -> None:
        super().__init__()
        check_stop_init(stop_init_logit)
        self.policy_mlp = policy_mlp
        self.use_bias = use_bias
        # Zeros, so at init the stop logit is whatever the trained policy MLP makes
        # of an all-zero "option" -- a real function of the board, not a constant.
        self.stop_token = nn.Parameter(torch.zeros(d_model))
        self.stop_bias = nn.Parameter(
            torch.full((MAX_STOP_BIAS + 1,), float(stop_init_logit))
        )

    def forward(self, pooled_state: torch.Tensor) -> torch.Tensor:
        """(B, d) pooled state -> (B, MAX_STOP_BIAS + 1) stop logit per k."""
        batch = pooled_state.shape[0]
        token = self.stop_token[None, :].expand(batch, -1)
        joined = torch.cat([token, pooled_state], dim=-1)
        base = self.policy_mlp(joined).squeeze(-1)
        if not self.use_bias:
            return base[:, None].expand(batch, MAX_STOP_BIAS + 1)
        return base[:, None] + self.stop_bias[None, :]


@torch.no_grad()
def calibrate_stop_bias(
    stop_head: StopHead,
    pooled_states: torch.Tensor,
    option_logits: torch.Tensor,
    option_valid: torch.Tensor,
    margin: float = TIE_MARGIN,
) -> dict[str, float]:
    """Set ``stop_bias`` so greedy decoding provably ignores STOP on this sample.

    The champion takes the top-``maxCount`` options, so the tie holds iff the stop
    logit is below the *lowest* option logit it would have taken. We measure the
    minimum legal option logit per state, take the worst case over the sample, and
    place the stop logit ``margin`` below it.

    Returns the measured gap distribution so the calibration is reported rather
    than assumed. G-TIE then re-checks on a HELD-OUT sample.
    """
    if pooled_states.shape[0] != option_logits.shape[0]:
        raise GateRefusal("pooled states and option logits disagree on batch size")
    if not bool(option_valid.any()):
        raise GateRefusal("calibration sample has no legal options")

    neg_inf = torch.finfo(option_logits.dtype).max
    masked = option_logits.masked_fill(~option_valid, neg_inf)
    lowest = masked.min(dim=-1).values           # per state, the least-preferred option

    stop_head.stop_bias.zero_()
    base = stop_head(pooled_states)[:, 0]        # the un-biased stop logit
    gap = lowest - base                          # bias must be below this, per state
    worst = float(gap.min())
    bias = worst - margin
    check_stop_init(bias)
    stop_head.stop_bias.fill_(bias)

    return {
        "stop_bias": bias,
        "worst_gap": worst,
        "median_gap": float(gap.median()),
        "margin": margin,
        "n_states": int(pooled_states.shape[0]),
    }


def new_parameters(stop_head: StopHead) -> list[torch.nn.Parameter]:
    """The parameters that did not exist in the parent checkpoint.

    Returned separately so the runner can give them their own learning rate: they
    start at an arbitrary calibrated point while every other tensor starts at a
    trained optimum, and moving both at the same rate wastes the parent.
    """
    return [stop_head.stop_token, stop_head.stop_bias]


def describe(arch: Architecture) -> str:
    return (f"d{arch.d_model} · {arch.n_layers}L · {arch.n_heads} heads · "
            f"head_dim {arch.head_dim} · FFN {arch.ffn_hidden} · "
            f"numeric {arch.numeric_width} · cards {arch.card_vocab} · "
            f"relations {arch.n_relations} · value {arch.n_wdl}-class · "
            f"aux {list(arch.aux_classes)}")
