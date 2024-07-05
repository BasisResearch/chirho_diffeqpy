#!/bin/bash
set -euxo pipefail

./scripts/lint.sh
pytest -s --cov=chirho_diffeqpy/ --cov=tests --cov-report=term-missing ${@-} --cov-report html
