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
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "sklearn.utils._param_validation.Interval",
                    "sklearn.utils._param_validation.StrOptions",
                ],
            )

            results[name] = {
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "best_params": search.best_params_,
            }
            print(f"{name}: ROC-AUC {roc_auc:.4f}, F1 {f1:.4f}")

            if roc_auc > best["roc_auc"]:
                best = {
                    "roc_auc": roc_auc,
                    "pipeline": best_est,
                    "name": name,
                    "params": search.best_params_,
                }

    # Сохранение лучшей модели и метрик
    os.makedirs(args.models_dir, exist_ok=True)
    joblib.dump(best["pipeline"], os.path.join(args.models_dir, "model.joblib"))

    roc_path = os.path.join(args.models_dir, "roc_curve.png")
    proba_best = best["pipeline"].predict_proba(X_test)[:, 1]
    roc_artifact(y_test, proba_best, roc_path)

    metrics = {
        "best_model": best["name"],
        "roc_auc": best["roc_auc"],
        "precision": results[best["name"]]["precision"],
        "recall": results[best["name"]]["recall"],
        "f1": results[best["name"]]["f1"],
        "all_experiments": results,
    }
    metrics_path = os.path.join(args.models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Лучшая модель: {best['name']} (ROC-AUC {best['roc_auc']:.4f})")
    print(f"Метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()
