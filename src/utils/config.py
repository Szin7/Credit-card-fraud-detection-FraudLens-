"""Central configuration: non-secret tunables from config/settings.yaml,
secrets and environment-specific values from .env / the process environment.

Importing this module loads settings once into the module-level `settings`
object, so the rest of the codebase does `from src.utils.config import settings`
instead of re-reading YAML or re-parsing env vars in every module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class RiskBands:
    """UI risk categories — deliberately separate from the model's decision
    threshold (see Settings.decision_threshold). A transaction can be shown
    as HIGH risk for review without the binary fraud/not-fraud call changing.
    """

    low_max: float
    review_max: float

    def classify(self, probability: float) -> str:
        if probability < self.low_max:
            return "LOW"
        if probability < self.review_max:
            return "REVIEW"
        return "HIGH"


@dataclass(frozen=True)
class CostModel:
    """Simplified, illustrative cost simulation — not real business figures."""

    false_negative_cost: float
    false_positive_cost: float

    def expected_cost(self, false_negatives: int, false_positives: int) -> float:
        return (
            false_negatives * self.false_negative_cost
            + false_positives * self.false_positive_cost
        )


@dataclass(frozen=True)
class Settings:
    random_seed: int
    raw_dir: Path
    processed_dir: Path
    rejected_dir: Path
    model_dir: Path
    champion_filename: str
    metadata_filename: str
    decision_threshold: float
    risk_bands: RiskBands
    cost_model: CostModel
    max_duplicate_ratio_warning: float
    storage_mode: str
    raw_data_filename: str
    aws_region: str
    aws_s3_bucket: str
    database_url: str

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.champion_filename

    @property
    def model_metadata_path(self) -> Path:
        return self.model_dir / self.metadata_filename

    @property
    def raw_data_path(self) -> Path:
        return self.raw_dir / self.raw_data_filename


def _database_url() -> str:
    override = os.environ.get("DATABASE_URL")
    if override:
        return override
    user = os.environ.get("POSTGRES_USER", "fraudlens")
    password = os.environ.get("POSTGRES_PASSWORD", "fraudlens")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "fraudlens")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"


def load_settings() -> Settings:
    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        random_seed=int(raw["random_seed"]),
        raw_dir=PROJECT_ROOT / raw["paths"]["raw_dir"],
        processed_dir=PROJECT_ROOT / raw["paths"]["processed_dir"],
        rejected_dir=PROJECT_ROOT / raw["paths"]["rejected_dir"],
        model_dir=PROJECT_ROOT / raw["paths"]["model_dir"],
        champion_filename=raw["model"]["champion_filename"],
        metadata_filename=raw["model"]["metadata_filename"],
        decision_threshold=float(raw["model"]["decision_threshold"]),
        risk_bands=RiskBands(
            low_max=float(raw["risk_bands"]["low_max"]),
            review_max=float(raw["risk_bands"]["review_max"]),
        ),
        cost_model=CostModel(
            false_negative_cost=float(raw["cost_model"]["false_negative_cost"]),
            false_positive_cost=float(raw["cost_model"]["false_positive_cost"]),
        ),
        max_duplicate_ratio_warning=float(raw["validation"]["max_duplicate_ratio_warning"]),
        storage_mode=os.environ.get("STORAGE_MODE", "local"),
        raw_data_filename=os.environ.get("RAW_DATA_FILENAME", "creditcard_sample.csv"),
        aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        aws_s3_bucket=os.environ.get("AWS_S3_BUCKET", ""),
        database_url=_database_url(),
    )


settings = load_settings()
