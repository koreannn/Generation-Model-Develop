#!/bin/bash
set -euo pipefail

##################### 스크립트 파일 위치 기준 프로젝트 루트로 이동 후 커맨드 실행 #####################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

##################### 설정 변수 #####################
GIT_NAME="koreannn"
GIT_EMAIL="ghdtjdwo5@gmail.com"
PYTHON_VER="3.11"
USE_UV=true

##################### 로그 함수 #####################
log()  { echo -e "\n\033[1;32m[INFO]\033[0m $1"; }
warn() { echo -e "\n\033[1;33m[WARN]\033[0m $1"; }
die()  { echo -e "\n\033[1;31m[ERROR]\033[0m $1"; exit 1; }

##################### git 설정 #####################
log "git 전역 설정 중..."
git config --global user.name  "$GIT_NAME"
git config --global user.email "$GIT_EMAIL"
git config --global core.editor vim
git config --global init.defaultBranch main
git config --global credential.helper "cache --timeout=21600"

##################### .env파일 생성 #####################
log ".env 파일 설정 중..."

SAMPLE_ENV_PATH="$PROJECT_ROOT/config/sample/.env.sample"
TGT_ENV_PATH="$PROJECT_ROOT/config/.env"

if [ ! -f ".env" ]; then
    if [ -f "$SAMPLE_ENV_PATH" ]; then
        log "$SAMPLE_ENV_PATH 포맷을 읽어 .env 파일을 생성합니다."
        cp "$SAMPLE_ENV_PATH" "$TGT_ENV_PATH" 
        log ".env 파일 생성 완료."
    else
        warn "$SAMPLE_ENV_PATH 파일이 존재하지 않아 .env를 생성하지 못했습니다."
    fi
else
    warn ".env 파일이 이미 존재하므로 덮어쓰지 않고 기존 설정을 유지합니다."
fi

echo "----------------------------------------"
log "세팅 완료. 아래 명령어를 실행하십시오."
echo ""
echo "    source ~/.bashrc"
echo ""