"""Basic tests for data_preprocessing.CrimeDataPreprocessor."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import CrimeDataPreprocessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def processor():
    return CrimeDataPreprocessor()


@pytest.fixture
def missing_persons_df():
    """Two rows with every source column set to 1, so every derived total
    is a small, easy-to-check number."""
    cols = [
        "male_below_5_years", "male_5_to_14_years", "male_14_to_18_years",
        "male_18_to_30_years", "male_30_to_45_years", "male_45_to_60_years",
        "male_60_years_and_above",
        "female_below_5_years", "female_5_to_14_years", "female_14_to_18_years",
        "female_18_to_30_years", "female_30_to_45_years", "female_45_to_60_years",
        "female_60_years_and_above",
        "trangender_below_5_years", "trangender_5_to_14_years",
        "trangender_14_to_18_years", "trangender_18_to_30_years",
        "trangender_30_to_45_years", "trangender_45_to_60_years",
        "transgender_60_years_and_above",
    ]
    data = {c: [1, 1] for c in cols}
    data["year"] = [2018, 2019]
    data["state_name"] = ["StateA", "StateA"]
    data["district_name"] = ["Dist1", "Dist1"]
    return pd.DataFrame(data)


@pytest.fixture
def juvenile_crimes_df():
    crime_cols = [
        "murder", "attempt_to_commit_muder", "acid_attack", "atmpt_acid_attack",
        "rape", "attempt_to_rape", "sexual_harassment_at_work",
        "assault_on_women", "insult_women_modesty",
        "dacoity", "robbery", "burglary", "theft", "auto_theft",
    ]
    data = {c: [1, 1] for c in crime_cols}
    data["id"] = [1, 2]
    data["year"] = [2018, 2019]
    data["state_name"] = ["StateA", "StateA"]
    data["state_code"] = ["S1", "S1"]
    data["district_name"] = ["Dist1", "Dist1"]
    data["district_code"] = ["D1", "D1"]
    data["registration_circles"] = ["RC1", "RC1"]
    return pd.DataFrame(data)


def test_load_data_reads_committed_csvs():
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        processor = CrimeDataPreprocessor()
        missing_df, crimes_df = processor.load_data()
    finally:
        os.chdir(cwd)

    assert len(missing_df) > 0
    assert len(crimes_df) > 0
    for col in ("year", "state_name", "district_name"):
        assert col in missing_df.columns
        assert col in crimes_df.columns


def test_clean_missing_persons_computes_totals(processor, missing_persons_df):
    cleaned = processor.clean_missing_persons(missing_persons_df)

    assert (cleaned["male_total"] == 7).all()
    assert (cleaned["female_total"] == 7).all()
    assert (cleaned["transgender_total"] == 7).all()
    assert (cleaned["total_missing"] == 21).all()
    assert (cleaned["children_missing"] == 4).all()
    assert (cleaned["youth_missing"] == 4).all()
    assert (cleaned["adults_missing"] == 4).all()
    assert (cleaned["elderly_missing"] == 2).all()


def test_clean_missing_persons_fills_na(processor, missing_persons_df):
    missing_persons_df.loc[0, "male_below_5_years"] = np.nan
    cleaned = processor.clean_missing_persons(missing_persons_df)
    assert not cleaned["male_below_5_years"].isna().any()


def test_clean_juvenile_crimes_computes_totals(processor, juvenile_crimes_df):
    cleaned = processor.clean_juvenile_crimes(juvenile_crimes_df)

    assert (cleaned["violent_crimes"] == 4).all()  # murder, attempt_to_commit_muder, acid_attack, atmpt_acid_attack
    assert (cleaned["sexual_crimes"] == 5).all()
    assert (cleaned["property_crimes"] == 5).all()
    assert (cleaned["total_crimes"] == 14).all()


def test_merge_datasets_inner_join(processor, missing_persons_df, juvenile_crimes_df):
    missing_clean = processor.clean_missing_persons(missing_persons_df)
    crimes_clean = processor.clean_juvenile_crimes(juvenile_crimes_df)

    merged = processor.merge_datasets(missing_clean, crimes_clean)

    assert len(merged) == 2
    assert "total_missing" in merged.columns
    assert "total_crimes" in merged.columns


def test_merge_datasets_drops_non_matching_keys(processor, missing_persons_df, juvenile_crimes_df):
    missing_clean = processor.clean_missing_persons(missing_persons_df)
    crimes_clean = processor.clean_juvenile_crimes(juvenile_crimes_df)
    crimes_clean.loc[:, "district_name"] = "SomewhereElse"

    merged = processor.merge_datasets(missing_clean, crimes_clean)

    assert len(merged) == 0


def test_create_features_adds_ratios(processor, missing_persons_df, juvenile_crimes_df):
    missing_clean = processor.clean_missing_persons(missing_persons_df)
    crimes_clean = processor.clean_juvenile_crimes(juvenile_crimes_df)
    merged = processor.merge_datasets(missing_clean, crimes_clean)

    featured = processor.create_features(merged)

    assert (featured["year_numeric"] == featured["year"] - 2017).all()
    assert "missing_to_crime_ratio" in featured.columns
    assert "gender_ratio" in featured.columns
    # male_total == female_total == 7 in the fixture, so ratio is exactly 1.
    assert np.allclose(featured["gender_ratio"], 7 / (7 + 1))


def test_get_state_aggregated_sums_by_group(processor):
    df = pd.DataFrame({
        "year": [2018, 2018, 2019],
        "state_name": ["StateA", "StateA", "StateA"],
        "total_crimes": [10, 20, 5],
        "state_code": [1, 1, 1],
        "district_code": [1, 2, 1],
    })

    agg = processor.get_state_aggregated(df)

    row_2018 = agg[(agg["year"] == 2018) & (agg["state_name"] == "StateA")]
    assert row_2018["total_crimes"].iloc[0] == 30
    row_2019 = agg[(agg["year"] == 2019) & (agg["state_name"] == "StateA")]
    assert row_2019["total_crimes"].iloc[0] == 5


def test_prepare_for_modeling_drops_missing_target(processor):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [4, 5, 6],
        "target": [10, np.nan, 30],
    })

    X, y, feature_cols = processor.prepare_for_modeling(df, "target")

    assert len(X) == 2
    assert len(y) == 2
    assert "target" not in feature_cols
    assert set(feature_cols) == {"feature_a", "feature_b"}


def test_prepare_for_modeling_respects_explicit_feature_cols(processor):
    df = pd.DataFrame({
        "feature_a": [1, 2, 3],
        "feature_b": [4, 5, 6],
        "target": [10, 20, 30],
    })

    X, y, feature_cols = processor.prepare_for_modeling(df, "target", feature_cols=["feature_a"])

    assert feature_cols == ["feature_a"]
    assert list(X.columns) == ["feature_a"]
