# PD Scoring Pipeline

[![CI](https://github.com/FurtiveDev/Automation_hw/actions/workflows/ci.yml/badge.svg)](https://github.com/FurtiveDev/Automation_hw/actions/workflows/ci.yml)

Итоговый проект по дисциплине "Автоматизация процессов разработки и тестирования моделей машинного обучения".

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
