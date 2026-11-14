#!/usr/bin/env bash
# Сборка Docker-образа скорингового API
set -e
DD_CLI="${DOCKER_CLI:-docker}"
"$DD_CLI" build -t pd-scoring-api .
echo "Образ собран: pd-scoring-api"
