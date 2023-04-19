#!/bin/bash

if [ "$RUN_JUPYTERLAB" = "True" ]; then
  pip install jupyterlab
  pip install ipywidgets nbformat matplotlib plotly
  jupyter lab --ip 0.0.0.0 --port 8889 --notebook-dir=/home/tf_docker/user_code
elif [ "$RUN_MARKET_SERVER" = "True" ]; then
  python dedicated_market_server.py
else
  python run_tradeforce_default.py
fi
