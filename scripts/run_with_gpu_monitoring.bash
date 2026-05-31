#!/bin/bash

# adamw_torch 설정
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free --format=csv -l 1 > ../log/gpu_log_adamw_torch.csv &
NVIDIA_LOG_PID_1=$!
nohup python -u main.py --config config-adamw_torch.yaml &
PYTHON_PID_1=$!
wait $PYTHON_PID_1

# 첫 번째 모니터링 프로세스 종료
kill $NVIDIA_LOG_PID_1

# adafactor 설정
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free --format=csv -l 1 > ../log/gpu_log_adafactor.csv &
NVIDIA_LOG_PID_2=$!
nohup python -u main.py --config config-adafactor.yaml &
PYTHON_PID_2=$!
wait $PYTHON_PID_2

# 두 번째 모니터링 프로세스 종료
kill $NVIDIA_LOG_PID_2

# adamw_bnb_8bit 설정
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free --format=csv -l 1 > ../log/gpu_log_adamw_bnb_8bit.csv &
NVIDIA_LOG_PID_3=$!
nohup python -u main.py --config config-adamw_bnb_8bit.yaml &
PYTHON_PID_3=$!
wait $PYTHON_PID_3

# 세 번째 모니터링 프로세스 종료
kill $NVIDIA_LOG_PID_3
