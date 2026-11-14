# -*- coding: utf-8 -*-
"""Unit-тесты feature engineering на синтетических данных."""
import pandas as pd
import pytest

from src.features.build_features import add_features


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "LIMIT_BAL": [200000, 50000],
            "SEX": [2, 1],
            "EDUCATION": [2, 3],
            "MARRIAGE": [1, 2],
            "AGE": [24, 55],
            "PAY_0": [0, 2],
            "PAY_2": [-1, 2],
            "PAY_3": [0, 3],
            "PAY_4": [0, 2],
            "PAY_5": [-1, 2],
            "PAY_6": [0, 3],
            "BILL_AMT1": [10000, 1000],
            "BILL_AMT2": [9000, 900],
            "BILL_AMT3": [8000, 800],
            "BILL_AMT4": [7000, 700],
            "BILL_AMT5": [6000, 600],
            "BILL_AMT6": [5000, 500],
            "PAY_AMT1": [1000, 100],
            "PAY_AMT2": [1000, 100],
            "PAY_AMT3": [1000, 100],
            "PAY_AMT4": [1000, 100],
            "PAY_AMT5": [1000, 100],
            "PAY_AMT6": [1000, 100],
        }
    )


def test_add_features_columns(raw_df):
    out = add_features(raw_df)
    for col in ["pay_mean_delay", "pay_max_delay", "pay_late_count",
                "bill_total", "pay_total", "pay_to_bill_ratio", "age_group"]:
        assert col in out.columns


def test_pay_late_count(raw_df):
    out = add_features(raw_df)
    # У первого клиента нет задержек, у второго задержка во всех шести месяцах
    assert out["pay_late_count"].tolist() == [0, 6]


def test_pay_mean_delay(raw_df):
    out = add_features(raw_df)
    assert out["pay_mean_delay"].iloc[0] == pytest.approx((0 - 1 + 0 + 0 - 1 + 0) / 6)
    assert out["pay_mean_delay"].iloc[1] == pytest.approx((2 + 2 + 3 + 2 + 2 + 3) / 6)


def test_pay_to_bill_ratio(raw_df):
    out = add_features(raw_df)
    # Первый клиент: платежей 6000, счетов 45000
    assert out["pay_to_bill_ratio"].iloc[0] == pytest.approx(6000 / (45000 + 1))


def test_age_group_bins(raw_df):
    out = add_features(raw_df)
    assert out["age_group"].iloc[0] == "21-30"
    assert out["age_group"].iloc[1] == "51-60"
