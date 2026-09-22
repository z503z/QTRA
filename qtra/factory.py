"""Explicit model construction and checkpoint loading."""
import json
from pathlib import Path

import torch


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def build_model(config, clip_pretrained, x3d_pretrained=None):
    from .models.baseline import Change3D_QAG_Simple
    from .models.cdvqa import QChange3D_SharedQRACalibration_CDVQA
    from .models.qag import (
        QChange3D_QAG_SharedQRA,
        QChange3D_QAG_SharedQRAAnswerCalibration,
        QChange3D_QAG_SharedQRAAnswerCalibrationDualGate,
        QChange3D_QAG_SharedQRAAnswerCalibrationDualGateMaskGuided,
    )

    cfg = dict(config)
    variant = cfg.pop("variant")
    common = dict(
        clip_pretrained=str(clip_pretrained),
        change3d_pretrained=str(x3d_pretrained) if x3d_pretrained else None,
        num_answers=cfg.pop("num_answers"),
        image_size=cfg.pop("image_size"),
    )
    constructors = {
        "baseline": Change3D_QAG_Simple,
        "qtra": QChange3D_QAG_SharedQRA,
        "global": QChange3D_QAG_SharedQRAAnswerCalibration,
        "grounding": QChange3D_QAG_SharedQRAAnswerCalibrationDualGate,
        "full": QChange3D_QAG_SharedQRAAnswerCalibrationDualGateMaskGuided,
        "cdvqa": QChange3D_SharedQRACalibration_CDVQA,
    }
    if variant not in constructors:
        raise ValueError(f"Unknown variant: {variant}")
    return constructors[variant](**common, **cfg)


def read_checkpoint(path):
    # Use restricted loading. Do not unpickle arbitrary objects from a download.
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def load_weights(model, path, allowed_missing=()):
    checkpoint = read_checkpoint(path)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    missing = [key for key in result.missing_keys if not key.startswith(tuple(allowed_missing))]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={missing}, unexpected={result.unexpected_keys}"
        )
    return checkpoint
