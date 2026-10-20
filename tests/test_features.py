import pandas as pd
import pytest

from core.features import build_feature_matrix


def _users(**overrides) -> pd.DataFrame:
    cols = {
        "user_id": ["a", "b"],
        "signup_date": ["2023-01-11", "2023-02-01"],
        "historical_spend": [10.0, 20.0],
        "region": ["north", "west"],
        "device_type": ["mobile", "tablet"],
    }
    cols.update(overrides)
    return pd.DataFrame(cols)


def test_columns_are_fixed_regardless_of_which_categories_are_present():
    features = build_feature_matrix(_users())
    assert list(features.columns) == [
        "historical_spend", "signup_days_since_start",
        "region_north", "region_south", "region_east", "region_west",
        "device_mobile", "device_desktop", "device_tablet",
    ]
    assert features["signup_days_since_start"].tolist() == [10, 31]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"region": ["north", "NORTH"]}, "outside"),
        ({"device_type": ["mobile", "phablet"]}, "outside"),
        ({"region": ["north", None]}, "missing"),
        ({"historical_spend": [1.0, float("nan")]}, "missing"),
    ],
)
def test_unencodable_inputs_raise_instead_of_becoming_all_zero_rows(overrides, match):
    with pytest.raises(ValueError, match=match):
        build_feature_matrix(_users(**overrides))
