#!/usr/bin/env bash
# SPI-1(30일) 단일 일별 윈도우 재학습 — 가뭄·복합 (폭염은 SPI 무관·기존 모델 재사용).
# 🔄 2026-06-03 SPI-3→SPI-1 전환. 입력 윈도우 30일 일별(멀티스케일 폐기).
# 손실·변수·분할 확정 유지. FD 버그는 train.py set_sharing_strategy로 이미 수정.
# set -e 미사용: 한 타깃 실패가 나머지를 죽이지 않게(독립 실행·exit 로깅).
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc   # ← SPI-1 라벨(별도 파일)

run() {  # $1=config $2=target $3=out
  echo "===== [$2] $(basename $1) 시작 $(date '+%F %T') ====="
  conda run -n cecd python -u -m src.train --config "$1" \
    --proc "$PROC" --labels "$LABELS" --full --target "$2" \
    --out "$3" > "logs/$(basename ${3%.pt}).log" 2>&1
  echo "===== [$2] 종료(exit=$?) $(date '+%F %T') ====="
}

run config/model_cnn_lstm_spi1.yaml     drought  checkpoints/cnn_lstm_drought_spi1.pt
run config/model_cnn_lstm_spi1.yaml     compound checkpoints/cnn_lstm_compound_spi1.pt
run config/model_transformer_spi1.yaml  compound checkpoints/transformer_compound_spi1.pt
echo "===== SPI-1 재학습 전체 완료 $(date '+%F %T') ====="
