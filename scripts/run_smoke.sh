#!/usr/bin/env bash
# Stage 0 얇은 수직 슬라이스: 다운로드 → 전처리 → 라벨링 → CNN-LSTM smoke
# 파이프라인 전체 관통 + 과적합(=학습 가능 신호) 확인.
set -euo pipefail
cd "$(dirname "$0")/.."

MONTH="${1:-2015-07}"
source /home/woobin/anaconda3/etc/profile.d/conda.sh

echo "=== [1/4] download $MONTH (data env) ==="
conda run -n data python -m src.data.download_era5 --month "$MONTH"

echo "=== [2/4] preprocess $MONTH (cecd env) ==="
conda run -n cecd python -m src.data.preprocess --month "$MONTH"

echo "=== [3/4] labeling $MONTH (cecd env) ==="
conda run -n cecd python -m src.labeling.compound --tag "$MONTH"

echo "=== [4/4] CNN-LSTM smoke (cecd env) ==="
# 1개월 슬라이스는 SPI-3 불가 → compound 전부 0 → heatwave 라벨로 신호 확인
conda run -n cecd python -m src.train \
  --config config/model_cnn_lstm.yaml \
  --proc "data/processed/era5_daily_${MONTH}.nc" \
  --labels "data/labels/labels_${MONTH}.nc" \
  --target heatwave --smoke

echo "=== 게이트 0: 위에서 train_loss 하강 + train CSI 상승(과적합) 확인 ==="
