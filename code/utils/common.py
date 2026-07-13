import argparse
from contextlib import contextmanager
from datetime import datetime
import os
import random
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from loguru import logger
import numpy as np
import torch
import yaml


# 코드 전역에서 첫 실행 시점의 타임스탬프를 동일하게 사용
CURRENT_TIME = None


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(os.path.join("../config", args.config), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_env_file(filepath="../config/.env"):
    try:
        # .env 파일 로드 시도
        if load_dotenv(filepath):
            logger.debug(f".env 파일을 성공적으로 로드했습니다: {filepath}")
        else:
            raise FileNotFoundError  # 파일이 없으면 예외 발생
    except FileNotFoundError:
        logger.debug(f"경고: 지정된 .env 파일을 찾을 수 없습니다: {filepath}")
    except Exception as e:
        logger.debug(f"오류 발생: .env 파일 로드 중 예외가 발생했습니다: {e}")


def set_logger(log_file="../log/file.log", log_level="DEBUG"):
    # 로거 설정
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        level=log_level,
        rotation="12:00",  # 매일 12시에 새로운 로그 파일 생성
        retention="7 days",  # 7일 후 로그 제거
    )


# config 확인
def log_config(config, depth=0):
    if depth == 0:
        print("*" * 40)
    for k, v in config.items():
        prefix = ["\t" * depth, k, ":"]

        if isinstance(v, dict):
            print(*prefix)
            log_config(v, depth + 1)
        else:
            prefix.append(v)
            print(*prefix)
    if depth == 0:
        print("*" * 40)


def get_current_time():
    global CURRENT_TIME
    if CURRENT_TIME is None:
        CURRENT_TIME = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%m%d%H%M")
    return CURRENT_TIME


def create_experiment_filename(config):
    if config is None:
        config = load_config()
    username = config["exp"]["username"]
    base_model = config["model"]["base_model"].replace("/", "_")
    train_path = config["data"]["train_path"]
    train_name = os.path.splitext(os.path.basename(train_path))[0]
    num_train_epochs = config["training"]["params"]["num_train_epochs"]
    learning_rate = config["training"]["params"]["learning_rate"]
    current_time = get_current_time()

    return f"{username}_{base_model}_{train_name}_{num_train_epochs}_{learning_rate}_{current_time}"


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logger.debug(f"[{name}] done in {time.time() - t0:.3f} s")
