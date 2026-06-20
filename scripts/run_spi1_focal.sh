#!/usr/bin/env bash
# 🔬 손실 통일 실험(A): CNN-LSTM 복합을 Focal로 재학습 (Transformer는 이미 Focal·재사용).
# 현 정식 cnn_lstm_compound_spi1.pt(BCE)는 보존, 출력은 *_spi1_focal.pt 별도 태그.
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc

echo "===== [compound] CNN-LSTM Focal 재학습 시작 $(date '+%F %T') ====="
"$CONDA" run --no-capture-output -n cecd python -u -m src.train \
  --config config/model_cnn_lstm_spi1_focal.yaml \
  --proc "$PROC" --labels "$LABELS" --full --target compound \
  --out checkpoints/cnn_lstm_compound_spi1_focal.pt \
  > logs/cnn_lstm_compound_spi1_focal.log 2>&1
echo "===== [compound] 종료(exit=$?) $(date '+%F %T') ====="
