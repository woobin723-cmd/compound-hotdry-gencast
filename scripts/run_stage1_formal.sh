#!/usr/bin/env bash
# Stage 1 정식 재학습: 예측성(CSI) 극대화 설정.
#   개선점(확정 손실 유지): 양성-윈도우 오버샘플 · cosine+warmup LR · 조기종료 · grad clip
#                          · pos_weight 캡(CNN-LSTM) · dropout/wd 강화(Transformer)
#   확정 손실 변경 없음: Transformer=Focal(0.25,2.0), CNN-LSTM=Weighted BCE.
#   조기종료가 best-val 가중치를 저장하므로 epoch 상한은 넉넉히 둠(낭비 없음).
#
# ⚠️ 컴퓨터 재시작 후 실행. 종료 후 PC를 끌 예정이면 학습 완료 로그를 먼저 확인할 것.
# 실행:  nohup bash scripts/run_stage1_formal.sh > logs/run_formal.out 2>&1 &
set -u
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PROC=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_1989-2020.nc
mkdir -p logs checkpoints

echo "===== [1/2] CNN-LSTM(정식) 시작 $(date '+%F %T') ====="
conda run -n cecd python -u -m src.train --config config/model_cnn_lstm.yaml \
  --proc "$PROC" --labels "$LABELS" --full --target compound \
  --out checkpoints/cnn_lstm_formal.pt > logs/train_cnn_lstm_formal.log 2>&1
echo "===== [1/2] CNN-LSTM 종료(exit=$?) $(date '+%F %T') ====="

echo "===== [2/2] Transformer(정식) 시작 $(date '+%F %T') ====="
conda run -n cecd python -u -m src.train --config config/model_transformer.yaml \
  --proc "$PROC" --labels "$LABELS" --full --target compound \
  --out checkpoints/transformer_formal.pt > logs/train_transformer_formal.log 2>&1
echo "===== [2/2] Transformer 종료(exit=$?) $(date '+%F %T') ====="

echo "===== [3/3] 앙상블 평가(CNN-LSTM + Transformer) $(date '+%F %T') ====="
conda run -n cecd python -u -m scripts.ensemble_eval \
  --proc "$PROC" --labels "$LABELS" \
  --ckpts checkpoints/cnn_lstm_formal.pt checkpoints/transformer_formal.pt \
  > logs/ensemble_eval.log 2>&1
echo "===== 정식 재학습 전체 완료 $(date '+%F %T') ====="
