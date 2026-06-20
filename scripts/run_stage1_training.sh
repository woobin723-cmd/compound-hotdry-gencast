#!/usr/bin/env bash
# Stage 1 본학습: CNN-LSTM(베이스라인) → Transformer(주력) 순차 실행.
# target=compound, --full(연도 분할), 로그는 unbuffered(python -u)로 실시간 기록.
# 12h 예산: CNN-LSTM 15ep(~7.5h) + Transformer 40ep(~2h).
set -u
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_1989-2020.nc

echo "===== [1/2] CNN-LSTM 시작 $(date '+%F %T') ====="
conda run -n cecd python -u -m src.train --config config/model_cnn_lstm.yaml \
  --proc "$PROC" --labels "$LABELS" --full --target compound \
  --out checkpoints/cnn_lstm_full.pt > logs/train_cnn_lstm_full.log 2>&1
echo "===== [1/2] CNN-LSTM 종료(exit=$?) $(date '+%F %T') ====="

echo "===== [2/2] Transformer 시작 $(date '+%F %T') ====="
conda run -n cecd python -u -m src.train --config config/model_transformer.yaml \
  --proc "$PROC" --labels "$LABELS" --full --target compound \
  --out checkpoints/transformer_full.pt > logs/train_transformer_full.log 2>&1
echo "===== [2/2] Transformer 종료(exit=$?) $(date '+%F %T') ====="
echo "===== Stage 1 학습 전체 완료 $(date '+%F %T') ====="
