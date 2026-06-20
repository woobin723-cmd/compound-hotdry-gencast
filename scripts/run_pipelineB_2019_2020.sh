#!/usr/bin/env bash
# 엄밀화①: GenCast 학습기간(1979–2018) 밖인 2019–2020만으로 Pipeline B 재확인.
# 결론(가뭄≫복합≈폭염·리드한계)이 학습겹침 없는 연도에서도 유지되는지 검증.
set -uo pipefail
cd /data/woobin/deeplearning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CONDA=/home/woobin/anaconda3/condabin/conda
ERA5=data/processed/era5_daily_1989-2020.nc
LABELS=data/labels/labels_spi1_1989-2020.nc
for TGT in heatwave compound drought; do
  echo "===== [2019-2020] $TGT $(date '+%T') ====="
  "$CONDA" run --no-capture-output -n cecd python -u -m src.eval.skill_decay \
    --mode leadtime --target $TGT --year 2019 2020 \
    --era5 "$ERA5" --labels "$LABELS" --manifest config/detectors_spi1.json \
    --n-leads 15 --save-json "data/skill_json/leadtime_${TGT}_1920.json" \
    > "logs/pipelineB_${TGT}_1920.log" 2>&1
  echo "===== [2019-2020] $TGT 종료(exit=$?) ====="
done
echo "===== ✅ 2019-2020 재확인 완료 $(date '+%T') ====="
