#!/usr/bin/env bash
# 🔬 폭염 doy-아노말리 4멤버 학습(2026-06-14, B안): 입력에 t2m_anom(평년대비 편차) 채널 추가
#   (in_vars 5)해 폭염 라벨(doy-p95) 정의와 입력을 정렬. 기존 4채널 폭염 모델은 그대로 보존,
#   별도 태그 *_heatwave_anom_*. proc=era5_daily_anom(5채널). 단독 Test 지표 로그 출력→ablation.
#   set -e 미사용(멤버 독립). 실패 누적→nonzero 종료.
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda

PROC=data/processed/era5_daily_anom_1989-2020.nc   # 5채널(t2m,tp,Z500,t850,t2m_anom)
LABELS=data/labels/labels_spi1_1989-2020.nc        # 폭염 라벨(불변)

FAILED=0
run_member() {  # $1=model $2=config $3=out
  echo "===== [heatwave_anom] $1 학습 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.train \
    --config "$2" --proc "$PROC" --labels "$LABELS" --full --target heatwave \
    --out "$3" > "logs/$(basename ${3%.pt}).log" 2>&1
  local rc=$?
  echo "===== [heatwave_anom] $1 종료(exit=$rc) $(date '+%F %T') ====="
  [ "$rc" -ne 0 ] && FAILED=$((FAILED + 1)); return 0
}

run_member unet        config/model_unet_anom_spi1_focal.yaml      checkpoints/unet_heatwave_anom_spi1_focal.pt
run_member convlstm    config/model_convlstm_anom_spi1_focal.yaml  checkpoints/convlstm_heatwave_anom_spi1_focal.pt
run_member cnn_lstm    config/model_cnn_lstm_anom_spi1_focal.yaml  checkpoints/cnn_lstm_heatwave_anom_spi1_focal.pt
run_member transformer config/model_transformer_anom_spi1.yaml     checkpoints/transformer_heatwave_anom_spi1.pt

if [ "$FAILED" -ne 0 ]; then
  echo "===== ⚠️ 폭염 아노말리 학습 종료 — 실패 ${FAILED}개 $(date '+%F %T') ====="; exit 1
fi
echo "===== ✅ 폭염 doy-아노말리 4멤버 학습 완료(실패 0) $(date '+%F %T') ====="
