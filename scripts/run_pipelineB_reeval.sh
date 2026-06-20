#!/usr/bin/env bash
# Pipeline B 재평가(2026-06-15): detector 교체 반영(폭염·복합 아노말리 5채널·가뭄 4멤버).
# GenCast 원본 재추론 불필요 — 후처리 nc 그대로, 하이브리드 윈도우에서 t2m_anom 즉석 산출.
# 6년×15init=90 init·8멤버. 결과 JSON은 leadtime_{target}.json(그림 재생성용).
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONDA=/home/woobin/anaconda3/condabin/conda
ERA5=data/processed/era5_daily_1989-2020.nc     # 과거 채움(4채널·anom은 하이브리드서 즉석)
LABELS=data/labels/labels_spi1_1989-2020.nc
YEARS="2015 2016 2017 2018 2019 2020"

for TGT in heatwave compound drought; do
  echo "===== [Pipeline B] $TGT 재평가 시작 $(date '+%F %T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.eval.skill_decay \
    --mode leadtime --target $TGT --year $YEARS \
    --era5 "$ERA5" --labels "$LABELS" --manifest config/detectors_spi1.json \
    --n-leads 15 --save-json "data/skill_json/leadtime_${TGT}.json" \
    > "logs/pipelineB_${TGT}.log" 2>&1
  echo "===== [Pipeline B] $TGT 종료(exit=$?) $(date '+%F %T') ====="
done
echo "===== ✅ Pipeline B 재평가 3타깃 완료 $(date '+%F %T') ====="
