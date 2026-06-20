#!/usr/bin/env bash
# 🔬 폭염·가뭄 4멤버 앙상블 확장(2026-06-13): 복합과 동일 구조(30일 일별·Focal 통일)로
#   폭염·가뭄 각각 4멤버(CNN-LSTM·ConvLSTM·U-Net·Transformer) 순차 학습.
#   기존 단일 모델(cnn_lstm_drought_spi1.pt=BCE·cnn_lstm_heatwave.pt=14일)은 보존, 별도 태그.
#   폭염이 복합 병목 → 폭염 먼저. 각 단독 Test 지표가 로그에 출력 → 이후 ablation으로 조합 선정.
#   set -e 미사용: 한 멤버 실패가 나머지를 죽이지 않게(독립 실행·exit 로깅).
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
CONDA=/home/woobin/anaconda3/condabin/conda

PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc   # SPI-1 라벨(폭염=SPI 무관·동일 라벨 내 heatwave 변수)

# 멤버: 약칭 → (config, 출력 접미사). transformer만 *_spi1(focal 내장), 나머지 *_spi1_focal.
FAILED=0  # 실패 멤버 누적(전부 끝난 뒤 nonzero로 종료 — codex 지적: 실패가 성공으로 기록되던 문제)
run_member() {  # $1=target $2=model $3=config $4=out
  echo "===== [$1] $2 Focal 학습 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.train \
    --config "$3" --proc "$PROC" --labels "$LABELS" --full --target "$1" \
    --out "$4" > "logs/$(basename ${4%.pt}).log" 2>&1
  local rc=$?
  echo "===== [$1] $2 종료(exit=$rc) $(date '+%F %T') ====="
  [ "$rc" -ne 0 ] && FAILED=$((FAILED + 1))
  return 0   # 한 멤버 실패가 루프를 끊지 않게(독립 실행)
}

# 타깃 순서: 폭염(병목·우선) → 가뭄. 멤버 순서: 빠른 것 먼저(U-Net→ConvLSTM→CNN-LSTM→Transformer).
for TGT in heatwave drought; do
  run_member "$TGT" unet        config/model_unet_spi1_focal.yaml      "checkpoints/unet_${TGT}_spi1_focal.pt"
  run_member "$TGT" convlstm    config/model_convlstm_spi1_focal.yaml  "checkpoints/convlstm_${TGT}_spi1_focal.pt"
  run_member "$TGT" cnn_lstm    config/model_cnn_lstm_spi1_focal.yaml  "checkpoints/cnn_lstm_${TGT}_spi1_focal.pt"
  run_member "$TGT" transformer config/model_transformer_spi1.yaml     "checkpoints/transformer_${TGT}_spi1.pt"
done
if [ "$FAILED" -ne 0 ]; then
  echo "===== ⚠️ 폭염·가뭄 학습 종료 — 실패 멤버 ${FAILED}개 $(date '+%F %T') ====="
  exit 1
fi
echo "===== ✅ 폭염·가뭄 4멤버 학습 전체 완료(실패 0) $(date '+%F %T') ====="
