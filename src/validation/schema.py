"""Canonical transaction schema, shared by ingestion (structural check) and
validation (full rule checks) so the column list only lives in one place.
"""

V_FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS = ["Time", *V_FEATURE_COLUMNS, "Amount", "Class"]
INFERENCE_COLUMNS = ["Time", *V_FEATURE_COLUMNS, "Amount"]
NUMERIC_COLUMNS = ["Time", "Amount", *V_FEATURE_COLUMNS]
VALID_CLASS_VALUES = {0, 1}
