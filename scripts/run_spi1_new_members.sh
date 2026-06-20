#!/usr/bin/env bash
# 🔬 앙상블 멤버 확장(2026-06-08): ConvLSTM·U-Net 복합 멤버를 Focal로 순차 학습.
#   기존 cnn_lstm/transformer compound 멤버와 손실 통일(Focal). 각 단독 Test 지표가 로그에
#   출력됨 → 이후 ablation으로 최적 앙상블 조합 선정. 현 정식 가중치는 전부 보존, 별도 태그.
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc

for MODEL in convlstm unet; do
  echo "===== [compound] ${MODEL} Focal 학습 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.train \
    --config "config/model_${MODEL}_spi1_focal.yaml" \
    --proc "$PROC" --labels "$LABELS" --full --target compound \
    --out "checkpoints/${MODEL}_compound_spi1_focal.pt" \
    > "logs/${MODEL}_compound_spi1_focal.log" 2>&1
  echo "===== [compound] ${MODEL} 종료(exit=$?) $(date '+%F %T') ====="
done
