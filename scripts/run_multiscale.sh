#!/usr/bin/env bash
# 멀티스케일 다운샘플 윈도우(25스텝/90일) CNN-LSTM 학습.
# 가뭄·복합 향상이 목표(폭염은 14일로 이미 양호 → 제외). best-val 가중치 자동 저장.
set -euo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_1989-2020.nc
CFG=config/model_cnn_lstm_ms.yaml

for TGT in drought compound; do
  echo "===== [$TGT] 멀티스케일 CNN-LSTM 시작 $(date '+%F %T') ====="
  conda run -n cecd python -u -m src.train --config "$CFG" \
    --proc "$PROC" --labels "$LABELS" --full --target "$TGT" \
    --out "checkpoints/cnn_lstm_${TGT}_ms.pt" > "logs/train_cnn_lstm_${TGT}_ms.log" 2>&1
  echo "===== [$TGT] 종료(exit=$?) $(date '+%F %T') ====="
done
echo "===== 멀티스케일 학습 전체 완료 $(date '+%F %T') ====="
