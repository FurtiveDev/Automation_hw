# -*- coding: utf-8 -*-
"""Этап 1 пайплайна (DVC): загрузка, очистка, валидация и feature engineering."""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.data.validation import validate_input  # noqa: E402
from src.features.build_features import add_features  # noqa: E402


def load_raw(raw_path: str) -> pd.DataFrame:
    """Читает UCI xls, приводит заголовки и целевую к аккуратному виду."""
    df = pd.read_excel(raw_path, header=1, engine="calamine")
    df = df.rename(columns={"default payment next month": "default"})
    df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)
    df = df.drop(columns=["ID"])

    # Привожу категории к справочным значениям из описания датасета
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    df = df.reset_index(drop=True)
    return df


def prepare(raw_path: str, out_path: str) -> pd.DataFrame:
    """Загрузка -> очистка -> валидация Pandera -> feature engineering -> сохранение."""
    df = load_raw(raw_path)

    # Валидация очищенных сырых данных (провалится на аномалиях)
    df = validate_input(df)

    df = add_features(df)
    df = df.convert_dtypes()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Подготовлено: {len(df)} строк, {df.shape[1]} колонок -> {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Этап prepare: загрузка, очистка, валидация, признаки"
    )
    parser.add_argument("--raw", default="data/raw/default of credit card clients.xls")
    parser.add_argument("--out", default="data/processed/processed.csv")
    args = parser.parse_args()
    prepare(args.raw, args.out)


if __name__ == "__main__":
    main()
