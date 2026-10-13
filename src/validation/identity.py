"""Stable fallback identity when an upstream transaction ID is unavailable."""

import hashlib
import json

from src.validation.schema import INFERENCE_COLUMNS


def feature_digest(row) -> str:
    values = [float(row[column]) for column in INFERENCE_COLUMNS]
    values = [0.0 if number == 0 else number for number in values]
    payload = json.dumps(values, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
