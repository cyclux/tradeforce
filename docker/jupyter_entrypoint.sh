#!/bin/bash

pip install jupyterlab
pip install ipywidgets nbformat matplotlib plotly
jupyter lab --ip 0.0.0.0 --port 8889 --notebook-dir=/home/tf_docker/user_code
