"""Shared utilities for DEF-aoexp."""

from __future__ import annotations

import json

import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and torch tensors."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return super().default(obj)
