#!/bin/bash
git pull
source activate.sh
python experiments/run_experiments_acflow.py -c experiments/configs/afa_configs/mnist_acflow.yaml --debug
