#!/bin/bash

# 첫 번째 실험
nohup python -u main.py --config config-normal.yaml &
PYTHON_PID_1=$!
wait $PYTHON_PID_1

# 두 번째 실험
nohup python -u main.py --config config-rag.yaml &
PYTHON_PID_2=$!
wait $PYTHON_PID_2
