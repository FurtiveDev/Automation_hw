# -*- coding: utf-8 -*-
"""Мониторинг дрифта: отправляет новые данные в API и считает PSI по ключевым признакам.

Запуск при работающем API (uvicorn или Docker):
    python scripts/drift_check.py --n 200

Скрипт берёт случайную часть данных, отправляет каждую строку на /predict,
собирает предсказанные вероятности дефолта и сравнивает распределение
с вероятностями на train-данных (эталон). Затем считает PSI по ключевым
признакам между train и новыми данными.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.features.build_features import add_features

KEY_FEATURES = ["LIMIT_BAL", "AGE", "pay_mean_delay", "bill_total", "pay_to_bill_ratio"]


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index между двумя распределениями."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_dist = np.histogram(expected, bins=edges)[0] / len(expected)
    a_dist = np.histogram(actual, bins=edges)[0] / len(actual)
    e_dist = np.clip(e_dist, 1e-6, None)
    a_dist = np.clip(a_dist, 1e-6, None)
    return float(np.sum((a_dist - e_dist) * np.log(a_dist / e_dist)))


def psi_report(reference_df: pd.DataFrame, new_df: pd.DataFrame, features: list) -> pd.DataFrame:
    rows = []
    for feature in features:
        value = psi(reference_df[feature].dropna().values, new_df[feature].dropna().values)
        level = "нет дрифта" if value < 0.1 else ("умеренный" if value < 0.25 else "сильный")
        rows.append({"feature": feature, "psi": round(value, 4), "уровень": level})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Скрипт мониторинга дрифта данных (PSI)")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", default="data/processed/processed.csv")
    parser.add_argument("--n", type=int, default=200, help="сколько новых строк отправить в API")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_df = df.iloc[: int(len(df) * 0.8)]
    new_df = df.iloc[-args.n:].reset_index(drop=True)
    new_df_fe = add_features(new_df)

    # Эталон: вероятности дефолта на train через обученную модель
    import joblib

    pipeline = joblib.load(os.path.join("models", "model.joblib"))
    ref_proba = pipeline.predict_proba(train_df.drop(columns=["default"]))[:, 1]

    # Отправляем новые данные в API и собираем вероятности дефолта
    print(f"Отправляю {len(new_df)} строк в {args.api}/predict ...")
    api_proba = []
    for _, row in new_df_fe.iterrows():
        payload = {col: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for col, v in row.items() if col in pd.read_csv(args.data, nrows=0).columns}
        resp = requests.post(f"{args.api}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        api_proba.append(data["probability"] if "probability" in data else 1 - data["probability_no_default"])
    api_proba = np.array(api_proba)

    report = psi_report(train_df.assign(probability=ref_proba), new_df.assign(probability=api_proba),
                        KEY_FEATURES + ["probability"])
    print(report.to_string(index=False))
    print("\nПороги PSI: < 0.1 нет дрифта, 0.1-0.25 умеренный, > 0.25 сильный")

    with open("drift_report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print("Отчёт сохранён: drift_report.json")


if __name__ == "__main__":
    main()
