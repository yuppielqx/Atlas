"""Swappable local encoders for CausalFedST clients.

Every encoder is an ``nn.Module`` with the same contract:

    __init__(input_dim, hid_dim, seq_len, num_layers, dropout, **model_specific)
    forward(x, temporal_id=None) -> z: (B, 1, hid_dim)

The downstream PrototypeMatching / Predictor / OT-Hungarian alignment / MI
regulariser only depend on ``hid_dim`` and the prototype tensor, never on the
encoder internals, so any module honouring this contract is a drop-in backbone.
Encoders must also be ``torch.func.vmap`` safe to run on the default vectorized
training path (no data-dependent control flow / dynamic shapes).
"""

import inspect

from models.encoder import Encoder as STIDEncoder
from .dlinear import DLinearEncoder
from .patchtst import PatchTSTEncoder
from .nlinear import NLinearEncoder
from .tsmixer import TSMixerEncoder
from .timesnet import TimesNetEncoder
from .autoformer import AutoformerEncoder
from .informer import InformerEncoder
from .gru import GRUEncoder
from .cnfgnn import CNFGNNEncoder

ENCODER_REGISTRY = {
    "stid": STIDEncoder,
    "dlinear": DLinearEncoder,
    "patchtst": PatchTSTEncoder,
    "nlinear": NLinearEncoder,
    "tsmixer": TSMixerEncoder,
    "timesnet": TimesNetEncoder,
    "autoformer": AutoformerEncoder,
    "informer": InformerEncoder,
    "gru": GRUEncoder,
    "cnfgnn": CNFGNNEncoder,
}

# Encoders whose original mechanisms (FFT + top-k period/delay selection,
# ProbSparse query sampling) use data-dependent control flow that breaks
# torch.func.vmap. They run only on the per-client (--vectorized false) path.
VMAP_UNSAFE = {"timesnet", "autoformer", "informer"}


def is_vmap_safe(encoder_type):
    """Whether this encoder can run on the default vectorized (vmap) path."""
    return encoder_type not in VMAP_UNSAFE

# Constructor args shared by every encoder (supplied by the trainer, not the
# user). Anything else in a given encoder's __init__ is a model-specific
# hyperparameter, surfaced via --encoder_opt.
_COMMON_ARGS = {
    "self",
    "input_dim",
    "hid_dim",
    "seq_len",
    "num_layers",
    "dropout",
    "multivariate",
}


def default_encoder_kwargs(encoder_type):
    """Model-specific hyperparameters and their defaults, read from the encoder.

    The single source of truth is each encoder's ``__init__`` signature, so the
    per-model config lives in ``models/encoders/<model>.py`` rather than in
    main.py. Returns e.g. ``{"patch_len": 4, "stride": 2, ...}`` for patchtst and
    ``{}`` for stid.
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder_type {encoder_type!r}; available: {sorted(ENCODER_REGISTRY)}"
        )
    sig = inspect.signature(ENCODER_REGISTRY[encoder_type].__init__)
    return {
        name: p.default
        for name, p in sig.parameters.items()
        if name not in _COMMON_ARGS and p.default is not inspect.Parameter.empty
    }


def _cast_like(default, value):
    """Cast a string CLI override to the type of the encoder default."""
    if isinstance(default, bool):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot parse bool from {value!r}")
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def build_encoder_kwargs(encoder_type, overrides=None):
    """Merge an encoder's defaults with ``KEY=VAL`` override strings.

    Args:
        encoder_type: registry key.
        overrides: iterable of ``"key=value"`` strings (e.g. from --encoder_opt).

    Raises:
        ValueError: on bad syntax or an unknown hyperparameter for this encoder.
    """
    kwargs = default_encoder_kwargs(encoder_type)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--encoder_opt expects KEY=VAL, got {item!r}")
        key, value = item.split("=", 1)
        if key not in kwargs:
            raise ValueError(
                f"Unknown {encoder_type} option {key!r}; "
                f"available: {sorted(kwargs) or '(none)'}"
            )
        kwargs[key] = _cast_like(kwargs[key], value)
    return kwargs


def build_encoder(
    encoder_type,
    *,
    input_dim,
    hid_dim,
    seq_len,
    num_layers,
    dropout,
    encoder_kwargs=None,
    multivariate=False,
):
    """Construct an encoder by name.

    Args:
        encoder_type: registry key, e.g. "stid" | "dlinear" | "patchtst".
        encoder_kwargs: optional dict of model-specific extra constructor args.
        multivariate: if True, all input channels are treated as series features
            (for multi-feature causal discovery datasets like KnowAir-Full).
            Only applies to the STID encoder.

    Raises:
        ValueError: if ``encoder_type`` is not registered.
    """
    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder_type {encoder_type!r}; "
            f"available: {sorted(ENCODER_REGISTRY)}"
        )
    cls = ENCODER_REGISTRY[encoder_type]
    merged_kwargs = dict(encoder_kwargs or {})
    if encoder_type == "stid":
        # Force-set multivariate — may override the encoder's own default
        # (which build_encoder_kwargs reads from the signature).
        merged_kwargs["multivariate"] = multivariate
    return cls(
        input_dim=input_dim,
        hid_dim=hid_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        dropout=dropout,
        **merged_kwargs,
    )


__all__ = [
    "ENCODER_REGISTRY",
    "VMAP_UNSAFE",
    "is_vmap_safe",
    "build_encoder",
    "build_encoder_kwargs",
    "default_encoder_kwargs",
    "STIDEncoder",
    "DLinearEncoder",
    "PatchTSTEncoder",
    "NLinearEncoder",
    "TSMixerEncoder",
    "TimesNetEncoder",
    "AutoformerEncoder",
    "InformerEncoder",
    "GRUEncoder",
    "CNFGNNEncoder",
]
