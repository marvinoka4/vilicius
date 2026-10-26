#!/bin/sh
# Convert the percent-format sources into Kaggle notebooks.
set -e
cd "$(dirname "$0")/.."
jupytext --to ipynb --set-kernel python3 notebooks/eda.py -o kaggle/eda/store-sales-eda.ipynb -q
# The pipeline notebook is src/pipeline.py (shared with the local experiments)
# followed by notebooks/pipeline_tail.py (validation, final fit, submission).
mkdir -p kaggle/pipeline
cat src/pipeline.py notebooks/pipeline_tail.py > outputs/pipeline_notebook.py
jupytext --to ipynb --set-kernel python3 outputs/pipeline_notebook.py -o kaggle/pipeline/store-sales-lightgbm.ipynb -q
jupytext --to ipynb --set-kernel python3 notebooks/backtesting.py -o kaggle/backtesting/backtesting-store-sales.ipynb -q
