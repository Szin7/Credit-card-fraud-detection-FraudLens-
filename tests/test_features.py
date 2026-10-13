import numpy as np
import pandas as pd

from src.features.build_features import build_feature_pipeline
from src.validation.schema import V_FEATURE_COLUMNS


def _sample_frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in V_FEATURE_COLUMNS}
    data["Time"] = rng.uniform(0, 172_792, size=n)
    data["Amount"] = rng.lognormal(mean=3.0, sigma=1.0, size=n)
    return pd.DataFrame(data)


def test_output_has_one_row_per_input_row():
    df = _sample_frame(15)
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)
    assert len(out) == 15


def test_output_column_count_matches_expected_features():
    # 1 amount + 2 cyclical time + 28 passthrough V columns
    df = _sample_frame(10)
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)
    assert out.shape[1] == 1 + 2 + len(V_FEATURE_COLUMNS)


def test_amount_is_log_transformed_and_scaled():
    df = _sample_frame(500)
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)

    amount_col = out["amount__Amount"]
    # Standard-scaled output should be roughly centered and unit-scale.
    assert abs(amount_col.mean()) < 0.3
    assert 0.5 < amount_col.std() < 2.0


def test_time_cyclical_features_are_bounded():
    df = _sample_frame(200)
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)

    assert out["time_cyclical__time_hour_sin"].between(-1.0, 1.0).all()
    assert out["time_cyclical__time_hour_cos"].between(-1.0, 1.0).all()


def test_midnight_boundary_is_continuous_in_cyclical_encoding():
    # Hour 23:59 and hour 00:01 should map close together in sin/cos space,
    # unlike a raw hour-of-day number where they're 23 apart.
    df = pd.DataFrame(
        {
            **{col: [0.0, 0.0] for col in V_FEATURE_COLUMNS},
            "Time": [23 * 3600 + 59 * 60, 1 * 60],
            "Amount": [10.0, 10.0],
        }
    )
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)

    sin_vals = out["time_cyclical__time_hour_sin"].to_numpy()
    cos_vals = out["time_cyclical__time_hour_cos"].to_numpy()
    distance = np.hypot(sin_vals[0] - sin_vals[1], cos_vals[0] - cos_vals[1])
    assert distance < 0.2


def test_pca_features_pass_through_unchanged():
    df = _sample_frame(10)
    pipeline = build_feature_pipeline()
    out = pipeline.fit_transform(df)

    np.testing.assert_allclose(out["pca_features__V1"].to_numpy(), df["V1"].to_numpy())


def test_pipeline_is_reproducible_on_refit():
    df = _sample_frame(50)
    out_a = build_feature_pipeline().fit_transform(df)
    out_b = build_feature_pipeline().fit_transform(df)
    pd.testing.assert_frame_equal(out_a, out_b)
