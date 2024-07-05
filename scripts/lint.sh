#!/bin/bash
set -euxo pipefail

mypy --ignore-missing-imports chirho/
isort --check --profile black --diff chirho_diffeqpy/ tests/
black --check chirho_diffeqpy/ tests/
flake8 chirho_diffeqpy/ tests/