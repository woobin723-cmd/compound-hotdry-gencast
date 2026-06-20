#!/usr/bin/env bash
# 🔬 복합 doy-아노말리 4멤버 학습(2026-06-15): heat 성분이 복합 병목 → 폭염에서 +60% 낸
#   t2m doy-아노말리 채널(in_vars5)을 복합에도 적용. 폭염과 동일 config·proc(era5_daily_anom),
#   타깃만 compound. 기존 4채널 복합 모델 보존, 별도 태그 *_compound_anom_*.
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda
PROC=data/processed/era5_daily_anom_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc

FAILED=0
run_member() {  # $1=model $2=config $3=out
  echo "===== [compound_anom] $1 학습 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.train \
    --config "$2" --proc "$PROC" --labels "$LABELS" --full --target compound \
    --out "$3" > "logs/$(basename ${3%.pt}).log" 2>&1
  local rc=$?
  echo "===== [compound_anom] $1 종료(exit=$rc) $(date '+%F %T') ====="
  [ "$rc" -ne 0 ] && FAILED=$((FAILED + 1)); return 0
}

run_member unet        config/model_unet_anom_spi1_focal.yaml      checkpoints/unet_compound_anom_spi1_focal.pt
run_member convlstm    config/model_convlstm_anom_spi1_focal.yaml  checkpoints/convlstm_compound_anom_spi1_focal.pt
run_member cnn_lstm    config/model_cnn_lstm_anom_spi1_focal.yaml  checkpoints/cnn_lstm_compound_anom_spi1_focal.pt
run_member transformer config/model_transformer_anom_spi1.yaml     checkpoints/transformer_compound_anom_spi1.pt

if [ "$FAILED" -ne 0 ]; then
  echo "===== ⚠️ 복합 아노말리 학습 종료 — 실패 ${FAILED}개 $(date '+%F %T') ====="; exit 1
fi
echo "===== ✅ 복합 doy-아노말리 4멤버 학습 완료(실패 0) $(date '+%F %T') ====="
