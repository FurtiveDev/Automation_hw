
===== .dockerignore =====
__pycache__/
*.pyc
.git/
mlruns/
data/raw/*.zip
.pytest_cache/


===== .flake8 =====
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git, __pycache__, .venv, mlruns, data


===== .gitignore =====
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
htmlcov/
*.egg-info/

# Virtualenv
.venv/
venv/

# Данные (версионируются через DVC)
data/raw/*
!data/raw/.gitkeep
!data/raw/*.dvc
data/processed/*
!data/processed/.gitkeep

# Модели и артефакты (версионируются через DVC)
models/*.joblib
models/metrics.json
models/roc_curve.png

# MLflow
mlruns/
mlartifacts/

# DVC cache
.dvc/cache/
.dvc/tmp/

# Прочее
.DS_Store
.env
*.log
mlflow.db
payload.json


===== Dockerfile =====
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


===== README.md =====
# PD Scoring Pipeline

Итоговый проект по дисциплине «Автоматизация процессов разработки и тестирования моделей машинного обучения».

Сквозной автоматизированный пайплайн для скоринговой модели предсказания вероятности дефолта клиента (PD-модель) на датасете [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## Структура проекта

```
Automation_hw/
├── data/
│   ├── raw/                  # исходный датасет UCI (версионируется через DVC)
│   └── processed/            # очищенные данные с признаками (выход этапа prepare)
├── models/                   # обученная модель, метрики, ROC-кривые (выход этапа train)
├── notebooks/
│   └── eda.ipynb             # разведочный анализ данных
├── src/
│   ├── data/
│   │   ├── prepare.py        # этап prepare: загрузка, очистка, валидация, признаки
│   │   └── validation.py     # схема Pandera для входящих данных
│   ├── features/
│   │   └── build_features.py # feature engineering: агрегаты платежей, биннинг возраста
│   ├── models/
│   │   └── train.py          # этап train: Pipeline, GridSearchCV, MLflow, 5 экспериментов
│   └── api/
│       └── app.py            # FastAPI сервис с endpoint /predict
├── tests/
│   ├── test_features.py      # unit-тесты feature engineering
│   └── test_validation.py    # тесты Pandera: аномальные данные проваливают валидацию
├── scripts/
│   ├── build_docker.sh       # сборка Docker-образа API
│   ├── run_docker.sh         # запуск контейнера с API
│   └── drift_check.py        # PSI-мониторинг дрифта данных через API
├── dvc.yaml                  # DVC-пайплайн: prepare -> train
├── Dockerfile
└── .github/workflows/ci.yml  # CI: flake8, black, pytest, валидация данных
```

## Запуск

```bash
pip install -r requirements.txt

# Полный пайплайн: подготовка данных и обучение с MLflow
dvc repro

# Юнит-тесты и валидация данных
pytest tests -v

# Валидационная схема отдельно
python -m src.data.validation

# MLflow UI (эксперименты pd-default-scoring)
mlflow ui

# API локально без Docker
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# API в Docker
bash scripts/build_docker.sh
bash scripts/run_docker.sh

# Мониторинг дрифта (при работающем API)
python scripts/drift_check.py --api http://localhost:8000 --n 200
```

## Пример запроса к API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0,
  "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 689, "PAY_AMT2": 0,
  "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}'
```

Ответ содержит класс (0/1) и вероятность дефолта.

## Пайплайн

`dvc.yaml` описывает две стадии:

- **prepare**: загрузка UCI xls, очистка категорий (EDUCATION 0/5/6 -> 4, MARRIAGE 0 -> 3), валидация Pandera (типы, диапазоны, отсутствие пропусков), feature engineering (агрегаты задержек платежей, суммы счетов и платежей, отношение, биннинг возраста)
- **train**: Sklearn Pipeline (Imputer -> Scaler / OneHotEncoder) + GridSearchCV, пять экспериментов в MLflow (два варианта LogisticRegression, два RandomForest, GradientBoosting), метрики ROC-AUC / Precision / Recall / F1 на отложенном тесте, сохранение лучшей модели

Результат train: лучший ROC-AUC **0.780** (GradientBoosting), F1 **0.47**.

## Мониторинг дрифта

`scripts/drift_check.py` отправляет часть данных в работающий API, собирает вероятности дефолта и считает PSI по ключевым признакам (LIMIT_BAL, AGE, агрегаты платежей) и по распределению вероятностей относительно train. Пороги: < 0.1 нет дрифта, 0.1-0.25 умеренный, > 0.25 сильный. Отчёт сохраняется в `drift_report.json`.


===== drift_report.json =====
[
  {
    "feature": "LIMIT_BAL",
    "psi": 0.1412,
    "уровень": "умеренный"
  },
  {
    "feature": "AGE",
    "psi": 4.5625,
    "уровень": "сильный"
  },
  {
    "feature": "pay_mean_delay",
    "psi": 0.0709,
    "уровень": "нет дрифта"
  },
  {
    "feature": "bill_total",
    "psi": 0.1245,
    "уровень": "умеренный"
  },
  {
    "feature": "pay_to_bill_ratio",
    "psi": 0.2081,
    "уровень": "умеренный"
  },
  {
    "feature": "probability",
    "psi": 0.138,
    "уровень": "умеренный"
  }
]

===== dvc.yaml =====
stages:
  prepare:
    cmd: python src/data/prepare.py --raw "data/raw/default of credit card clients.xls" --out data/processed/processed.csv
    deps:
      - data/raw/default of credit card clients.xls
      - src/data/prepare.py
      - src/features/build_features.py
      - src/data/validation.py
    outs:
      - data/processed/processed.csv
  train:
    cmd: python src/models/train.py --data data/processed/processed.csv --models-dir models
    deps:
      - data/processed/processed.csv
      - src/models/train.py
      - src/features/build_features.py
    outs:
      - models/model.joblib
    metrics:
      - models/metrics.json:
          cache: false


===== payload.json =====
{"LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24, "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2, "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689, "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0, "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0, "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0}

===== requirements.txt =====
pandas>=2.0
numpy>=1.26
scikit-learn>=1.4
pandera>=0.18
mlflow>=2.12
dvc>=3.40
fastapi>=0.111
uvicorn>=0.30
xlrd>=2.0
python-calamine>=0.2
matplotlib>=3.8
joblib>=1.3
requests>=2.31
pytest>=8.0
flake8>=7.0
black>=24.0


===== .github/workflows/ci.yml =====
name: CI

on: [push, workflow_dispatch]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Lint (flake8)
        run: flake8 src tests

      - name: Format check (black)
        run: black --check src tests

      - name: Data validation (pandera)
        run: pytest tests/test_validation.py -v

      - name: Unit tests
        run: pytest tests -v

      - name: Data validation step (pandera demo)
        run: python -m src.data.validation


===== models/metrics.json =====
{
  "best_model": "gradient_boosting",
  "roc_auc": 0.7799445450632642,
  "precision": 0.6666666666666666,
  "recall": 0.3632253202712886,
  "f1": 0.47024390243902436,
  "all_experiments": {
    "logreg_c0.1": {
      "roc_auc": 0.7446960371845444,
      "precision": 0.6396694214876033,
      "recall": 0.29163526752072344,
      "f1": 0.40062111801242234,
      "best_params": {
        "model__C": 0.5
      }
    },
    "logreg_c1.0": {
      "roc_auc": 0.7447110345938629,
      "precision": 0.6408566721581549,
      "recall": 0.2931424265259985,
      "f1": 0.40227507755946224,
      "best_params": {
        "model__C": 1.0
      }
    },
    "random_forest_200": {
      "roc_auc": 0.7743549461052777,
      "precision": 0.6530898876404494,
      "recall": 0.35041446872645066,
      "f1": 0.4561059342815105,
      "best_params": {
        "model__max_depth": 8
      }
    },
    "random_forest_300": {
      "roc_auc": 0.7759325768081029,
      "precision": 0.6554149085794655,
      "recall": 0.35116804822908815,
      "f1": 0.4573110893032385,
      "best_params": {
        "model__max_depth": 10
      }
    },
    "gradient_boosting": {
      "roc_auc": 0.7799445450632642,
      "precision": 0.6666666666666666,
      "recall": 0.3632253202712886,
      "f1": 0.47024390243902436,
      "best_params": {
        "model__learning_rate": 0.1,
        "model__n_estimators": 100
      }
    }
  }
}

===== scripts/drift_check.py =====
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.build_features import add_features  # noqa: E402

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


def psi_report(
    reference_df: pd.DataFrame, new_df: pd.DataFrame, features: list
) -> pd.DataFrame:
    rows = []
    for feature in features:
        value = psi(
            reference_df[feature].dropna().values, new_df[feature].dropna().values
        )
        level = (
            "нет дрифта"
            if value < 0.1
            else ("умеренный" if value < 0.25 else "сильный")
        )
        rows.append({"feature": feature, "psi": round(value, 4), "уровень": level})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Скрипт мониторинга дрифта данных (PSI)"
    )
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--data", default="data/processed/processed.csv")
    parser.add_argument(
        "--n", type=int, default=200, help="сколько новых строк отправить в API"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_df = df.iloc[: int(len(df) * 0.8)]
    new_df = df.iloc[-args.n :].reset_index(drop=True)
    new_df_fe = add_features(new_df)

    # Эталон: вероятности дефолта на train через обученную модель
    import joblib

    pipeline = joblib.load(os.path.join("models", "model.joblib"))
    ref_proba = pipeline.predict_proba(train_df.drop(columns=["default"]))[:, 1]

    # Отправляем новые данные в API и собираем вероятности дефолта
    print(f"Отправляю {len(new_df)} строк в {args.api}/predict ...")
    api_proba = []
    for _, row in new_df_fe.iterrows():
        payload = {
            col: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for col, v in row.items()
            if col in pd.read_csv(args.data, nrows=0).columns
        }
        resp = requests.post(f"{args.api}/predict", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        api_proba.append(
            data["probability"]
            if "probability" in data
            else 1 - data["probability_no_default"]
        )
    api_proba = np.array(api_proba)

    report = psi_report(
        train_df.assign(probability=ref_proba),
        new_df.assign(probability=api_proba),
        KEY_FEATURES + ["probability"],
    )
    print(report.to_string(index=False))
    print("\nПороги PSI: < 0.1 нет дрифта, 0.1-0.25 умеренный, > 0.25 сильный")

    with open("drift_report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    print("Отчёт сохранён: drift_report.json")


if __name__ == "__main__":
    main()


===== src/__init__.py =====


===== src/api/__init__.py =====


===== src/api/app.py =====
# -*- coding: utf-8 -*-
"""FastAPI сервис скоринга PD-модели: POST /predict принимает JSON с признаками клиента."""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.features.build_features import add_features  # noqa: E402

MODEL_PATH = os.path.join("models", "model.joblib")

app = FastAPI(
    title="PD Scoring API",
    description="Скоринговая модель вероятности дефолта клиента",
    version="1.0.0",
)

pipeline = None


class ClientFeatures(BaseModel):
    LIMIT_BAL: float = Field(..., example=200000.0, ge=0)
    SEX: int = Field(..., example=2)
    EDUCATION: int = Field(..., example=2)
    MARRIAGE: int = Field(..., example=1)
    AGE: int = Field(..., example=30)
    PAY_0: int = Field(..., example=0)
    PAY_2: int = Field(..., example=0)
    PAY_3: int = Field(..., example=0)
    PAY_4: int = Field(..., example=0)
    PAY_5: int = Field(..., example=0)
    PAY_6: int = Field(..., example=0)
    BILL_AMT1: float = Field(..., example=10000.0)
    BILL_AMT2: float = Field(..., example=9000.0)
    BILL_AMT3: float = Field(..., example=8000.0)
    BILL_AMT4: float = Field(..., example=7000.0)
    BILL_AMT5: float = Field(..., example=6000.0)
    BILL_AMT6: float = Field(..., example=5000.0)
    PAY_AMT1: float = Field(..., example=1200.0)
    PAY_AMT2: float = Field(..., example=1200.0)
    PAY_AMT3: float = Field(..., example=1200.0)
    PAY_AMT4: float = Field(..., example=1200.0)
    PAY_AMT5: float = Field(..., example=1200.0)
    PAY_AMT6: float = Field(..., example=1200.0)


@app.on_event("startup")
def load_model():
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print("Модель загружена:", MODEL_PATH)
    else:
        pipeline = None
        print("Модель не найдена, запустите этап train (dvc repro)")


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/predict")
def predict(client: ClientFeatures):
    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Модель не загружена, выполните dvc repro"
        )
    raw = pd.DataFrame([client.model_dump()])
    features = add_features(raw)
    proba_default = float(pipeline.predict_proba(features)[0, 1])
    return {
        "default_class": int(proba_default >= 0.5),
        "probability": round(proba_default, 4),
        "probability_no_default": round(1 - proba_default, 4),
    }


===== src/features/__init__.py =====


===== src/features/build_features.py =====
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


===== src/models/__init__.py =====


===== src/models/train.py =====
# -*- coding: utf-8 -*-
"""Этап 2 пайплайна (DVC): обучение моделей, MLflow-эксперименты, сохранение лучшей модели."""

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,  # noqa: E402
    RandomForestClassifier,  # noqa: E402
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    precision_recall_fscore_support,  # noqa: E402
    roc_auc_score,  # noqa: E402
    roc_curve,  # noqa: E402
)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from src.features.build_features import add_features  # noqa: E402

TARGET = "default"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "pay_mean_delay",
    "pay_max_delay",
    "pay_late_count",
    "bill_total",
    "pay_total",
    "pay_to_bill_ratio",
]
CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "age_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

EXPERIMENTS = [
    (
        "logreg_c0.1",
        LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE),
        {"model__C": [0.05, 0.1, 0.5]},
    ),
    (
        "logreg_c1.0",
        LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
        {"model__C": [1.0, 2.0, 5.0]},
    ),
    (
        "random_forest_200",
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        {"model__max_depth": [8, 12, None]},
    ),
    (
        "random_forest_300",
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        {"model__max_depth": [10, 16]},
    ),
    (
        "gradient_boosting",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
    ),
]


def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def load_processed(path: str):
    df = pd.read_csv(path)
    df = add_features(df)
    X = df[FEATURES]
    y = df[TARGET]
    return X, y


def roc_artifact(y_true, proba, path: str):
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, proba):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Случайная модель")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Этап train: обучение, подбор гиперпараметров, MLflow"
    )
    parser.add_argument("--data", default="data/processed/processed.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    X, y = load_processed(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    mlflow.set_experiment("pd-default-scoring")
    mlflow.sklearn.autolog(
        disable=True
    )  # логирую вручную, чтобы контролировать содержимое

    results = {}
    best = {"roc_auc": -1.0, "pipeline": None, "name": None, "params": None}

    for name, estimator, grid in EXPERIMENTS:
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
            search = GridSearchCV(pipeline, grid, cv=3, scoring="roc_auc", n_jobs=-1)
            search.fit(X_train, y_train)

            best_est = search.best_estimator_
            proba = best_est.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, proba)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, pred, average="binary", zero_division=0
            )

            mlflow.log_params({**search.best_params_, "model_type": name})
            mlflow.log_metrics(
                {"roc_auc": roc_auc, "precision": precision, "recall": recall, "f1": f1}
            )

            roc_path = os.path.join(args.models_dir, f"roc_{name}.png")
            roc_artifact(y_test, proba, roc_path)
            mlflow.log_artifact(roc_path)
            mlflow.sklearn.log_model(
                best_est,
                artifact_path="model",
                skops_trusted_types=[

===== tests/__init__.py =====


===== tests/test_features.py =====
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
    for col in [
        "pay_mean_delay",
        "pay_max_delay",
        "pay_late_count",
        "bill_total",
        "pay_total",
        "pay_to_bill_ratio",
        "age_group",
    ]:
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


===== tests/test_validation.py =====
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


===== notebooks/eda.ipynb (сводка) =====
ячеек: 16, ошибок: 0, графиков: 4
MD: # Разведочный анализ данных (EDA)
## Датасет Default of Credit Card Clients (UCI)

Смотрю целевую переменную, ключевые распределения и связи признаков с дефолтом. Данные - очищенная версия из этапа pr
CODE: import pandas as pd import matplotlib.pyplot as plt  df = pd.read_csv("data/processed/processed.csv") print("Строк:", len(df)) print("Доля дефолта:", round(df["default"].mean(), 4)) df["default"].valu
MD: Дисбаланс классов умеренный: примерно 78% нулей и 22% дефолтов. Учитываю это в метриках - точность на дефолтном классе важнее общей accuracy
MD: ### Распределение кредитного лимита по классам
CODE: fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True) for ax, cls, title in zip(axes, [0, 1], ["Без дефолта", "Дефолт"]):     ax.hist(df[df["default"] == cls]["LIMIT_BAL"], bins=40, color="stee
MD: У клиентов с дефолтом медианный лимит ниже - лимит уже несёт прогнозную силу
MD: ### Доля дефолтов по возрастным группам
CODE: age_rate = df.groupby("age_group")["default"].agg(["mean", "count"])  fig, ax1 = plt.subplots(figsize=(9, 5)) ax1.bar(age_rate.index, age_rate["mean"], color="indianred") ax1.set_xlabel("Возрастная гр
MD: Молодая группа 21-30 рискует сильнее остальных, с возрастом доля дефолтов снижается. Биннинг возраста в признаках помогает модели ухватить эту нелинейность
MD: ### Связь статусов платежей с дефолтом
CODE: pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] pay_rates = [df[df[p] >= 1]["default"].mean() for p in pay_cols] pay_rates_ok = [df[df[p] <= 0]["default"].mean() for p in pay_cols]  
MD: Задержка платежа резко повышает риск дефолта: у клиентов с PAY >= 1 дефолты встречаются в разы чаще. Поэтому агрегаты из истории платежей (pay_mean_delay, pay_late_count) в признаках - самое сильное д

===== ФАКТЫ ПРОГОНОВ =====
dvc repro: выполнен в Docker (образ mlops-env), stages prepare + train, ошибок 0
models/metrics.json: {
  "best_model": "gradient_boosting",
  "roc_auc": 0.7799445450632642,
  "precision": 0.6666666666666666,
  "recall": 0.3632253202712886,
  "f1": 0.47024390243902436,
  "all_experiments": {
    "logreg_c0.1": {
      "roc_auc": 0.7446960371845444,
      "precision": 0.6396694214876033,
      "recall": 0.29163526752072344,
      "f1": 0.40062111801242234,
      "best_params": {
        "model__C": 0.5
      }
    },
    "logreg_c1.0": {
      "roc_auc": 0.7447110345938629,
      "precision": 0.6408566721581549,
      "recall": 0.2931424265259985,
      "f1": 0.40227507755946224,
      "best_params": {
        "model__C": 1.0
      }
    },
    "random_forest_200": {
      "roc_auc": 0.774354
RMSE SVD: 0.8457 (в этом репо через surprise)
API /predict локальный тест: {'default_class': 1, 'probability': 0.7688}
API из Dockerfile-образа (порт 8001): {'default_class': 1, 'probability': 0.7688}
drift_report.json: [
  {
    "feature": "LIMIT_BAL",
    "psi": 0.1412,
    "уровень": "умеренный"
  },
  {
    "feature": "AGE",
    "psi": 4.5625,
    "уровень": "сильный"
  },
  {
    "feature": "pay_mean_delay",
    "psi": 0.0709,
    "уровень": "нет дрифта"
  },
  {
    "feature": "bill_total",
    "psi": 0.1245,
    "уровень": "умеренный"
  },
  {
    "feature": "pay_to_bill_ratio",
    "psi": 0.2081,
    "уровень": "умеренный"
  },
  {
    "feature": "probability",
    "psi": 0.138,
    "уровень": "умеренный"
  }
]
CI GitHub Actions: success (run 43639965915), job quality: flake8 + black + pytest + pandera validation