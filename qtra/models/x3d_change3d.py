"""X3D settings used by the perception-frame temporal encoder.

Use the upstream PyTorchVideo implementation directly. The temporal encoder
changes its default stem stride and disables the unused classification
activation; no vendored X3D implementation is needed.
"""
from pytorchvideo.models.x3d import create_x3d as _create_x3d


def create_x3d(**kwargs):
    kwargs.setdefault("stem_conv_stride", (1, 1, 1))
    kwargs.setdefault("head_activation", None)
    return _create_x3d(**kwargs)
