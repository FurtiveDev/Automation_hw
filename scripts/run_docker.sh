#!/usr/bin/env bash
# Запуск контейнера с API на порту 8000
set -e
DD_CLI="${DOCKER_CLI:-docker}"
"$DD_CLI" run --rm -p 8000:8000 --name pd-scoring-api pd-scoring-api
