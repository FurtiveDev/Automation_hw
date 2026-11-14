# -*- coding: utf-8 -*-
"""Инженерия признаков: агрегаты из истории платежей, биннинг возраста."""
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

AGE_BINS = [20, 30, 40, 50, 60, 100]
AGE_LABELS = ["21-30", "31-40", "41-50", "51-60", "60+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет агрегированные признаки из истории платежей и биннинг возраста."""
    out = df.copy()

    # Агрегаты по статусу задержек платежей за полгода
    out["pay_mean_delay"] = out[PAY_COLS].mean(axis=1)
    out["pay_max_delay"] = out[PAY_COLS].max(axis=1)
    out["pay_late_count"] = (out[PAY_COLS] > 0).sum(axis=1)

    # Суммы счетов и платежей, отношение
    out["bill_total"] = out[BILL_COLS].sum(axis=1)
    out["pay_total"] = out[PAY_AMT_COLS].sum(axis=1)
    out["pay_to_bill_ratio"] = out["pay_total"] / (out["bill_total"].abs() + 1)

    # Биннинг возраста
    out["age_group"] = pd.cut(
        out["AGE"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).astype(str)

    return out
