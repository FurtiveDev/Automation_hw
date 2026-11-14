# -*- coding: utf-8 -*-
"""Валидация входящих данных с помощью Pandera.

Схема описывает очищенные сырые данные: типы, отсутствие пропусков,
допустимые диапазоны значений. При нарушениях схема выбрасывает
SchemaError - тесты и пайплайн падают на аномальных данных.
"""
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors
import pandas as pd

RAW_COLUMNS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default",
]

raw_schema = pa.DataFrameSchema(
    {
        "LIMIT_BAL": pa.Column(float, pa.Check.in_range(0, 1_000_000), nullable=False),
        "SEX": pa.Column(int, pa.Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(int, pa.Check.isin([1, 2, 3, 4]), nullable=False),
        "MARRIAGE": pa.Column(int, pa.Check.isin([1, 2, 3]), nullable=False),
        "AGE": pa.Column(int, pa.Check.in_range(21, 79), nullable=False),
        "PAY_0": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "PAY_2": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "PAY_3": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "PAY_4": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "PAY_5": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "PAY_6": pa.Column(int, pa.Check.in_range(-2, 9), nullable=False),
        "default": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    },
    strict=False,
)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет очищенные сырые данные схемой. Возвращает DataFrame или SchemaError."""
    return raw_schema.validate(df, lazy=True)


if __name__ == "__main__":
    # Демо: синтетическая валидная выборка проходит проверку
    demo = pd.DataFrame(
        {
            "LIMIT_BAL": [50000.0, 200000.0],
            "SEX": [2, 1],
            "EDUCATION": [2, 3],
            "MARRIAGE": [1, 2],
            "AGE": [24, 45],
            "PAY_0": [0, -1],
            "PAY_2": [0, 2],
            "PAY_3": [-1, 0],
            "PAY_4": [0, 0],
            "PAY_5": [0, -1],
            "PAY_6": [-1, 2],
            "BILL_AMT1": [1000.0, 50000.0],
            "BILL_AMT2": [900.0, 49000.0],
            "BILL_AMT3": [800.0, 48000.0],
            "BILL_AMT4": [700.0, 47000.0],
            "BILL_AMT5": [600.0, 46000.0],
            "BILL_AMT6": [500.0, 45000.0],
            "PAY_AMT1": [100.0, 5000.0],
            "PAY_AMT2": [100.0, 5000.0],
            "PAY_AMT3": [100.0, 5000.0],
            "PAY_AMT4": [100.0, 5000.0],
            "PAY_AMT5": [100.0, 5000.0],
            "PAY_AMT6": [100.0, 5000.0],
            "default": [0, 1],
        }
    )
    validated = validate_input(demo)
    print(f"Демо-валидация прошла: {len(validated)} строк, схема корректна")
