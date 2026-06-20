#!/usr/bin/env bash
# SPI-1 복합 재학습(재시작으로 끊긴 2개만) — 드라우트는 완료되어 재사용.
# 🔄 2026-06-04 서버 재시작 복구. drought CNN-LSTM(Test CSI 0.5564)은 보존.
# --no-capture-output: conda run 버퍼링 해제 → 에폭 로그 실시간 기록(이전엔 버퍼링으로 끊김 시 로그 유실).
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda   # nohup 셸엔 conda PATH 없음 → 절대경로

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc   # SPI-1 라벨

run() {  # $1=config $2=target $3=out
  echo "===== [$2] $(basename $1) 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.train --config "$1" \
    --proc "$PROC" --labels "$LABELS" --full --target "$2" \
    --out "$3" > "logs/$(basename ${3%.pt}).log" 2>&1
  echo "===== [$2] 종료(exit=$?) $(date '+%F %T') ====="
}

run config/model_cnn_lstm_spi1.yaml     compound checkpoints/cnn_lstm_compound_spi1.pt
run config/model_transformer_spi1.yaml  compound checkpoints/transformer_compound_spi1.pt
echo "===== SPI-1 복합 재학습 전체 완료 $(date '+%F %T') ====="
