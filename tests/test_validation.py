# -*- coding: utf-8 -*-
"""Тест валидации данных: валидная выборка проходит, аномальная проваливается."""

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from src.data.validation import validate_input


def make_valid_row(age=30, limit=50000.0):
    row = {
        "LIMIT_BAL": limit,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": age,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "default": 0,
    }
    for i in range(1, 7):
        row[f"BILL_AMT{i}"] = 1000.0
        row[f"PAY_AMT{i}"] = 100.0
    return row


def valid_df(rows=3):
    return pd.DataFrame([make_valid_row() for _ in range(rows)])


def test_valid_data_passes():
    df = valid_df()
    assert len(validate_input(df)) == len(df)


def test_null_fails():
    df = valid_df()
    df.loc[0, "LIMIT_BAL"] = None
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_input(df)


def test_age_out_of_range_fails():
    df = valid_df()
    df.loc[0, "AGE"] = 500
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_input(df)


def test_negative_limit_fails():
    df = valid_df()
    df.loc[0, "LIMIT_BAL"] = -100
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_input(df)


def test_bad_education_fails():
    df = valid_df()
    df.loc[0, "EDUCATION"] = 7
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_input(df)


def test_pay_out_of_range_fails():
    df = valid_df()
    df.loc[0, "PAY_0"] = 99
    with pytest.raises((SchemaError, SchemaErrors)):
        validate_input(df)
