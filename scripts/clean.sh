#!/bin/bash
set -euxo pipefail

isort --profile black chirho_diffeqpy/ tests/
black chirho_diffeqpy/ tests/
