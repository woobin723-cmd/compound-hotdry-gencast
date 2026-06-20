#!/usr/bin/env bash
# 폭염·가뭄 단독 타깃 CNN-LSTM 학습 (게이트1 구성요소별 비교용).
# compound와 동일 하니스·동일 개선책, target만 변경. best-val 가중치 자동 저장.
set -euo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_1989-2020.nc

for TGT in heatwave drought; do
  echo "===== [$TGT] CNN-LSTM 시작 $(date '+%F %T') ====="
  conda run -n cecd python -u -m src.train --config config/model_cnn_lstm.yaml \
    --proc "$PROC" --labels "$LABELS" --full --target "$TGT" \
    --out "checkpoints/cnn_lstm_${TGT}.pt" > "logs/train_cnn_lstm_${TGT}.log" 2>&1
  echo "===== [$TGT] CNN-LSTM 종료(exit=$?) $(date '+%F %T') ====="
done
echo "===== 단독 타깃 학습 전체 완료 $(date '+%F %T') ====="
