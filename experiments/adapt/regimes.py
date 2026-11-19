"""Freeze and LoRA regimes for the guitar-transfer study.

Each regime is defined by the role a parameter plays rather than by module
name, so the same rule applies to models that share almost no structure:

    full           every parameter trains.
    probe          readouts only: the final projection of each supervised
                   output (a linear probe, Kumar et al., ICLR 2022, extended
                   to models with several outputs).
    lora           rank-8 adapters on every trunk matrix (linear, conv and
                   recurrent gate matrices, as QLoRA and ConvLoRA place them),
                   readouts trained in full, everything else frozen.
    lora_temporal  the same adapters on the temporal block only
                   (TEMPORAL_BLOCK): the BiLSTM / BiGRU layers of the recurrent
                   models, and the query and value projections of the
                   time-axis attention for the two transformers (Hu et al.,
                   ICLR 2022).

Readouts train in every regime, so the regimes differ only in what happens to
the trunk.

Roles, one map per model (OAF_ROLES, OAFV2_ROLES, HFT_ROLES, HPP_ROLES,
KONG_ROLES, TRANSKUN_ROLES; ROLES[name]):

    frontend  fixed feature extraction (STFT/CQT/mel filterbanks, TransKun's
              window); never trains
    encoder   everything that looks at the spectrogram before the model
              reasons along time
    temporal  the recurrent or attention stack that reasons along time
    readout   the final projection of each supervised output, one whose output
              the loss reads directly. A projection that only feeds a
              recurrent refinement stage (Kong's frame_model.fc and
              reg_onset_model.fc, OAFv2's frame_stack.1) is trunk, so the probe
              stays linear.

LoRA reaches every 2-D weight of Linear, Conv, ConvTranspose and LSTM/GRU gate
matrices, and the q/k/v weights of attention modules that hold them as bare
Parameters (TransKun's MultiHeadAttentionKernel, nn.MultiheadAttention). A
matrix whose smaller side is at most min_dim is frozen rather than adapted (a
rank-8 delta would outnumber its weights): for OAFv2, the four (48, 9) first
convolutions. Embedding tables and normalisation gains are frozen.

Only frozen BatchNorm is held in eval(); Dropout is left as each recipe sets it.

Two implementation details:

  1. nn.LSTM keeps its own _flat_weights list, which cuDNN reads instead of the
     module attributes, so a parametrization on weight_ih_l0 would be ignored
     in the forward pass. A forward pre-hook refreshes the list on every call;
     this costs the fused cuDNN kernel, so LoRA runs are slower.
  2. requires_grad = False does not stop a BatchNorm layer's running statistics
     from tracking the new data, because Lightning puts the model back in train
     mode at every epoch. Frozen BatchNorm leaves are put back in eval() after
     every model.train() (harden_frozen_norms). Only the leaves: a whole frozen
     block in eval() would switch OAFv2's BiLSTM to its chunked inference path.

Usage:

    import regimes as fr
    fr.apply(model, "lora", fr.OAFV2_ROLES)   # after the pretrained weights are loaded
    fr.harden_frozen_norms(model)

Merging a LoRA checkpoint into plain weights, from the state dict alone (every
model; Lightning's "state_dict" and TransKun's "state_dict" and
"best_state_dict"):

    python experiments/adapt/regimes.py merge in.ckpt out.ckpt [--alpha 16]
"""

import argparse
import fnmatch
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

REGIMES = ("full", "probe", "lora", "lora_temporal")

# Role -> module-name prefixes. Every parameter must match exactly one role or
# apply() raises, so no parameter trains (or stays frozen) by accident.
OAFV2_ROLES = {
    "frontend": ("mel",),
    "encoder": ("onset_stack.0", "offset_stack.0", "frame_stack.0", "velocity_stack.0"),
    # frame_stack.1 is the unsupervised "activation" feeding combined_stack's
    # BiLSTM; the frame loss reads combined_stack.1.
    "temporal": ("onset_stack.1", "offset_stack.1", "frame_stack.1", "combined_stack.0"),
    "readout": ("onset_stack.2", "offset_stack.2", "velocity_stack.1", "combined_stack.1"),
}

# Onsets and Frames (2018): the same four stacks as OAFv2 without the offset
# stack, and no in-model frontend -- the log-mel is read from the store.
OAF_ROLES = {
    "frontend": (),
    "encoder": ("onset_stack.0", "frame_stack.0", "velocity_stack.0"),
    # frame_stack.1 is the unsupervised "activation" feeding combined_stack's
    # BiLSTM; the frame loss reads combined_stack.1.
    "temporal": ("onset_stack.1", "frame_stack.1", "combined_stack.0"),
    "readout": ("onset_stack.2", "velocity_stack.1", "combined_stack.1"),
}

HFT_ROLES = {
    "frontend": (),
    "encoder": ("hft_encoder", "hft_decoder.pos_embedding_freq",
                "hft_decoder.layer_zero_freq", "hft_decoder.layers_freq"),
    "temporal": ("hft_decoder.pos_embedding_time", "hft_decoder.layers_time"),
    "readout": tuple(f"hft_decoder.fc_{head}_{axis}" for axis in ("freq", "time")
                     for head in ("onset", "offset", "frame", "velocity")),
}

# HPPNet-sp: two subnets, each a harmonic conv trunk and one BiLSTM + Linear head
# per output.  The CQT kernels are buffers, so the frontend holds no parameters.
_HPP_HEADS = (("subnet_onset", "onset"), ("subnet_frame", "frame"),
              ("subnet_frame", "offset"), ("subnet_frame", "velocity"))
HPP_ROLES = {
    "frontend": ("to_cqt", "amplitude_to_db"),
    "encoder": ("subnet_onset.trunk", "subnet_frame.trunk"),
    "temporal": tuple(f"{s}.heads.{h}.head.lstm" for s, h in _HPP_HEADS),
    "readout": tuple(f"{s}.heads.{h}.head.linear" for s, h in _HPP_HEADS),
}

# Kong: four acoustic models (conv blocks, fc5, a 2-layer BiGRU, an 88-way fc),
# then the onset and frame GRUs that combine them.  bn0 normalises the log-mel
# and trains in Kong's own recipe, so it belongs to the encoder, not the
# frontend (the STFT and mel weights are fixed buffers-as-parameters).
_KONG_ACOUSTIC = ("frame_model", "reg_onset_model", "reg_offset_model", "velocity_model")
KONG_ROLES = {
    "frontend": ("spectrogram_extractor", "logmel_extractor"),
    "encoder": ("bn0",) + tuple(f"{m}.{part}" for m in _KONG_ACOUSTIC
                                for part in ("conv1", "conv2", "conv3", "conv4",
                                             "fc5", "bnorm5")),
    # frame_model.fc and reg_onset_model.fc only feed frame_gru and reg_onset_gru
    # (the losses read frame_fc and reg_onset_fc); the offset and velocity
    # acoustic models' fc are supervised outputs.
    "temporal": tuple(f"{m}.gru" for m in _KONG_ACOUSTIC)
                + ("frame_model.fc", "reg_onset_model.fc", "reg_onset_gru", "frame_gru"),
    "readout": ("reg_offset_model.fc", "velocity_model.fc", "reg_onset_fc", "frame_fc"),
}

# TransKun 2.0 (the Transkun repository's model, pretrained/2.0.conf): a conv +
# time-frequency transformer backbone, then the semi-CRF interval scorer and the
# velocity and refined-onset/offset MLPs. As in every other model, only the last
# projection of each output is a readout; the hidden layer of the two output
# MLPs (768 -> 512) is trunk, so the probe is linear here too.
TRANSKUN_ROLES = {
    "frontend": ("framewiseFeatureExtractor",),
    "encoder": ("backbone",),
    "temporal": ("backbone.encoderLayers", "velocityPredictor.0", "refinedOFPredictor.0"),
    "readout": ("scorer", "velocityPredictor.3", "refinedOFPredictor.3"),
}

ROLES = {"oaf": OAF_ROLES, "oafv2": OAFV2_ROLES, "hft": HFT_ROLES, "hpp": HPP_ROLES,
         "kong": KONG_ROLES, "transkun": TRANSKUN_ROLES}

# Which roles train, per arm.  "frontend" never trains: for OAFv2 it holds no
# parameters at all (the mel is buffers), and where a model does put weights
# there they are a fixed filterbank.
TRAINABLE_ROLES = {
    "full": ("encoder", "temporal", "readout"),
    "probe": ("readout",),
    "lora": ("readout",),          # plus the adapters injected into the trunk
    "lora_temporal": ("readout",),  # plus the adapters injected into TEMPORAL_BLOCK
}

# Roles whose matrices receive adapters under the lora arm.
LORA_ROLES = ("encoder", "temporal")

# The temporal block of each model, for the lora_temporal arm: fnmatch patterns
# over full weight names (e.g. "onset_stack.1.rnn.weight_ih_l0").  Every match
# must also carry the "temporal" role; apply() checks that.  For the two
# transformers only the query and value projections of the time-axis attention
# are adapted (Hu et al. 2022, Sec. 7.1); keys, output projections and
# feed-forwards stay frozen.
TEMPORAL_BLOCK = {
    "oaf": ("onset_stack.1.*", "combined_stack.0.*"),
    "oafv2": ("onset_stack.1.*", "offset_stack.1.*", "combined_stack.0.*"),
    "hpp": tuple(f"{s}.heads.{h}.head.lstm.*" for s, h in _HPP_HEADS),
    "kong": tuple(f"{m}.gru.*" for m in _KONG_ACOUSTIC) + ("reg_onset_gru.*", "frame_gru.*"),
    "hft": ("hft_decoder.layers_time.*.self_attn.fc_query.weight",
            "hft_decoder.layers_time.*.self_attn.fc_value.weight"),
    "transkun": ("backbone.encoderLayers.*.mhaBlockT.module.q_proj_weight",
                 "backbone.encoderLayers.*.mhaBlockT.module.v_proj_weight"),
}


def _within(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def _model_name(roles: dict) -> str | None:
    return next((k for k, v in ROLES.items() if v is roles), None)


def _role_of(param_name: str, roles: dict) -> str | None:
    """Longest-prefix match, so 'frame_stack.1' wins over 'frame_stack'."""
    best, best_len = None, -1
    for role, prefixes in roles.items():
        for prefix in prefixes:
            if param_name == prefix or param_name.startswith(prefix + "."):
                if len(prefix) > best_len:
                    best, best_len = role, len(prefix)
    return best


class LoRADelta(nn.Module):
    """W -> W + (alpha/r) B @ A, registered as a parametrization on `weight`.

    B starts at zero so the adapted model is bit-identical to the pretrained one
    at step 0; A is Kaiming-uniform, as in Hu et al.  A conv kernel is treated
    as (out_channels, in_channels * kh * kw) and the delta is viewed back, which
    is the ConvLoRA formulation.
    """

    def __init__(self, shape: torch.Size, rank: int, alpha: float):
        super().__init__()
        out_features = shape[0]
        in_features = int(torch.tensor(shape[1:]).prod().item()) if len(shape) > 1 else 1
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self._shape = tuple(shape)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        delta = (self.B @ self.A) * self.scaling
        return weight + delta.view(self._shape)


def _refresh_flat_weights(module, _args):
    """Point nn.LSTM's cuDNN weight list at the parametrized attributes.

    See detail 1 in the module docstring: without it the adapters train and
    change nothing.
    """
    module._flat_weights = [
        getattr(module, name) if hasattr(module, name) else None
        for name in module._flat_weights_names
    ]


# Attention modules that keep their projections as bare Parameters rather than
# Linear children: nn.MultiheadAttention and TransKun's MultiHeadAttentionKernel.
_ATTENTION_PROJECTIONS = ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight")


def _adaptable(module: nn.Module):
    """Yield (weight_name, tensor) for every 2-D-or-more weight of `module`."""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        yield "weight", module.weight
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        for name in module._flat_weights_names:
            if name.startswith("weight_"):
                yield name, getattr(module, name)
    else:
        for name in _ATTENTION_PROJECTIONS:
            param = module._parameters.get(name)
            if isinstance(param, nn.Parameter) and param.dim() >= 2:
                yield name, param


def add_lora(model: nn.Module, roles: dict, rank: int = 8, alpha: float = 16.0,
             min_dim: int = 16, lora_roles=LORA_ROLES, within=None) -> dict:
    """Inject adapters into every eligible matrix of `lora_roles`. Returns a report.

    `within`, if given, restricts the matrices further to those whose full name
    matches one of its fnmatch patterns (the lora_temporal arm).
    """
    injected, skipped_small = [], []
    rnn_modules = set()
    # Both lists are materialised first: registering a parametrization adds
    # modules and moves parameters, which must not happen under a live iterator.
    for module_name, module in list(model.named_modules()):
        for weight_name, weight in list(_adaptable(module)):
            full_name = f"{module_name}.{weight_name}" if module_name else weight_name
            if _role_of(full_name, roles) not in lora_roles:
                continue
            if within is not None and not _within(full_name, within):
                continue
            out_features = weight.shape[0]
            in_features = int(torch.tensor(weight.shape[1:]).prod().item())
            if min(out_features, in_features) <= min_dim:
                skipped_small.append((full_name, tuple(weight.shape)))
                continue
            parametrize.register_parametrization(
                module, weight_name, LoRADelta(weight.shape, rank, alpha))
            injected.append((full_name, tuple(weight.shape),
                             rank * (out_features + in_features)))
            if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                rnn_modules.add((module_name, module))

    for _, module in rnn_modules:
        module.register_forward_pre_hook(_refresh_flat_weights)

    return {"injected": injected, "skipped_small": skipped_small,
            "n_rnn_patched": len(rnn_modules),
            "adapter_params": sum(n for _, _, n in injected)}


class FrozenNormEval(pl.Callback):
    """Hold frozen BatchNorm leaves in eval() for the whole run.

    See detail 2 in the module docstring.  Runs at every epoch start because
    Lightning calls model.train() there, undoing any earlier eval().
    """

    @staticmethod
    def _apply(model):
        for module in model.modules():
            # BatchNorm only: Dropout has no parameters and would always look frozen.
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                trainable = any(p.requires_grad for p in module.parameters(recurse=False))
                if not trainable:
                    module.eval()

    def on_train_epoch_start(self, trainer, pl_module):
        self._apply(pl_module)

    def on_train_start(self, trainer, pl_module):
        self._apply(pl_module)


def harden_frozen_norms(model: nn.Module) -> None:
    """Make `model.train()` itself re-freeze the norm leaves, on this instance.

    Equivalent to FrozenNormEval without access to the Trainer's callbacks.
    Lightning calls `model.train()` at every epoch start and after each
    validation; each call now ends by putting the frozen BatchNorm leaves back
    into eval().
    """
    original_train = model.train

    def train(mode: bool = True):
        original_train(mode)
        if mode:
            FrozenNormEval._apply(model)
        return model

    model.train = train  # type: ignore[method-assign]
    FrozenNormEval._apply(model)


def apply(model: nn.Module, regime: str, roles: dict, rank: int = 8,
          alpha: float = 16.0, min_dim: int = 16, verbose: bool = True) -> dict:
    """Put `model` into `regime`, after the pretrained weights are loaded.

    Returns a report dict and prints it, trainable parameter count included.
    """
    if regime not in REGIMES:
        raise SystemExit(f"unknown regime {regime!r}; one of {REGIMES}")

    unassigned = [n for n, _ in model.named_parameters() if _role_of(n, roles) is None]
    if unassigned:
        raise SystemExit(
            f"{len(unassigned)} parameters match no role, e.g. {unassigned[:5]}. "
            "The role map is out of date with the model -- fix it rather than "
            "letting them train or freeze by accident.")

    trainable_roles = TRAINABLE_ROLES[regime]
    for name, param in model.named_parameters():
        param.requires_grad = _role_of(name, roles) in trainable_roles

    lora_report = {}
    if regime == "lora":
        lora_report = add_lora(model, roles, rank, alpha, min_dim)
    elif regime == "lora_temporal":
        name = _model_name(roles)
        if name is None:
            raise SystemExit("lora_temporal needs one of the role maps in ROLES, "
                             "so the temporal block can be looked up")
        lora_report = add_lora(model, roles, rank, alpha, min_dim,
                               lora_roles=("temporal",), within=TEMPORAL_BLOCK[name])
        if not lora_report["injected"]:
            raise SystemExit(f"lora_temporal: no matrix of {name} matched TEMPORAL_BLOCK")
        # register_parametrization creates A/B with requires_grad=True; the
        # frozen original lives at ...parametrizations.<w>.original and must
        # stay frozen.
        for name, param in model.named_parameters():
            if ".parametrizations." in name and name.endswith(".original"):
                param.requires_grad = False

    by_role = {}
    for name, param in model.named_parameters():
        key = "adapter" if ".parametrizations." in name and not name.endswith(".original") \
            else (_role_of(name, roles) or "?")
        trained, total = by_role.setdefault(key, [0, 0])
        by_role[key] = [trained + (param.numel() if param.requires_grad else 0),
                        total + param.numel()]

    total = sum(t for _, t in by_role.values())
    trainable = sum(t for t, _ in by_role.values())
    report = {"regime": regime, "trainable": trainable, "total": total,
              "fraction": trainable / total, "by_role": by_role, **lora_report}

    if verbose:
        print(f"\n=== regime: {regime} ===")
        for role in ("frontend", "encoder", "temporal", "readout", "adapter"):
            if role in by_role:
                trained, tot = by_role[role]
                print(f"  {role:9s} {trained:>12,} / {tot:>12,} trainable")
        if lora_report:
            print(f"  adapters on {len(lora_report['injected'])} matrices "
                  f"({lora_report['adapter_params']:,} params), "
                  f"{len(lora_report['skipped_small'])} matrices left frozen as too small, "
                  f"{lora_report['n_rnn_patched']} RNN modules hooked")
            for name, shape in lora_report["skipped_small"]:
                print(f"    frozen (small): {name} {shape}")
        print(f"  TRAINABLE {trainable:,} of {total:,} = {100 * trainable / total:.2f}%\n")
    return report


def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its weight and drop the parametrizations.

    Do this before writing a checkpoint the evaluation scripts will read: they
    rebuild the architecture from the stored config and load strict, so adapter
    keys would be rejected.
    """
    merged = 0
    for module in model.modules():
        if parametrize.is_parametrized(module):
            for weight_name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, weight_name,
                                                    leave_parametrized=True)
                merged += 1
        if isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            module._flat_weights = [
                getattr(module, n) if hasattr(module, n) else None
                for n in module._flat_weights_names]
            module.flatten_parameters()
    return merged


def merge_state_dict(state_dict: dict, alpha: float = 16.0) -> tuple[dict, int]:
    """Fold LoRA parametrizations into plain weights, from the state dict alone.

    A parametrized weight appears as <module>.parametrizations.<w>.original,
    ...<w>.0.A and ...<w>.0.B; it becomes <module>.<w> = original + B@A*alpha/r
    with r read from A.  Needs no model class, so it serves every architecture.
    alpha must be the value the run trained with (the runs here use 16).
    """
    plain, parts = {}, {}
    for key, value in state_dict.items():
        if ".parametrizations." not in key:
            plain[key] = value
            continue
        prefix, rest = key.split(".parametrizations.", 1)
        weight_name, tail = rest.split(".", 1)
        parts.setdefault((prefix, weight_name), {})[tail] = value
    for (prefix, weight_name), p in parts.items():
        missing = {"original", "0.A", "0.B"} - set(p)
        if missing:
            raise SystemExit(f"{prefix}.{weight_name}: incomplete LoRA entry, missing {missing}")
        original, A, B = p["original"], p["0.A"], p["0.B"]
        delta = (B.double() @ A.double()) * (alpha / A.shape[0])
        plain[f"{prefix}.{weight_name}"] = (original.double() + delta.view(original.shape)).to(original.dtype)
    return plain, len(parts)


def merge_checkpoint_file(in_path: str, out_path: str, alpha: float = 16.0) -> int:
    """Merge every state dict in a checkpoint file and drop optimizer state."""
    ckpt = torch.load(in_path, map_location="cpu", weights_only=False)
    total = 0
    for key in ("state_dict", "best_state_dict"):
        if isinstance(ckpt.get(key), dict):
            ckpt[key], n = merge_state_dict(ckpt[key], alpha)
            total += n
    for key in ("optimizer_states", "lr_schedulers", "optimizer_state_dict",
                "lr_scheduler_state_dict"):
        ckpt.pop(key, None)
    torch.save(ckpt, out_path)
    print(f"merged {total} adapters; wrote {out_path}")
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("merge", help="fold LoRA adapters into a checkpoint's weights")
    m.add_argument("in_path")
    m.add_argument("out_path")
    m.add_argument("--alpha", type=float, default=16.0)
    args = ap.parse_args()
    if args.command == "merge":
        merge_checkpoint_file(args.in_path, args.out_path, args.alpha)
