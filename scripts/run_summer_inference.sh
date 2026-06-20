#!/bin/bash
# GenCast mini 여름철 15일 추론 — 2015~2020, 5일 간격 90 init
# 입력: /data/woobin/deeplearning/data/gencast_input/{year}/{MMDD}/
# 출력: /data/woobin/deeplearning/data/gencast_raw/{year}/{MMDD}/predictions.zarr
#
# raw_solar_vmap.py 입력 순서:
#   1) 모델 번호 (2 = GenCast 1p0deg Mini)
#   2) pressure-level 경로
#   3) single-instant 경로
#   4) single-accum 경로
#   5) ensemble_number (8)
#   6) predict steps (days) → 15
#   7) 출력 디렉토리

cd /data/woobin/gencast
source /home/woobin/anaconda3/etc/profile.d/conda.sh
conda activate gencast

export TMPDIR=/data/woobin/tmp
mkdir -p /data/woobin/tmp

export XLA_FLAGS="--xla_gpu_persistent_cache_dir=/data/woobin/gencast/.xla_cache"
mkdir -p /data/woobin/gencast/.xla_cache

INPUT_ROOT=/data/woobin/deeplearning/data/gencast_input
OUTPUT_ROOT=/data/woobin/deeplearning/data/gencast_raw

INIT_DATES="0615 0620 0625 0630 0705 0710 0715 0720 0725 0730 0804 0809 0814 0819 0824"
YEARS="2015 2016 2017 2018 2019 2020"

ENSEMBLE=8
DAYS=15   # 15일 = 30 step

start_total=$(date +%s)
echo "[$(date)] GenCast mini 여름철 추론 시작 (90 init, 15일×8ensemble)"

count=0
for year in $YEARS; do
  for mmdd in $INIT_DATES; do
    count=$((count+1))
    in_dir="$INPUT_ROOT/$year/$mmdd"
    out_dir="$OUTPUT_ROOT/$year/$mmdd"

    p="$in_dir/pressure_level.nc"
    i="$in_dir/single_instant.nc"
    a="$in_dir/single_accum.nc"

    echo "========================================"
    echo "[$(date)] [$count/90] $year/$mmdd"
    echo "========================================"

    # 입력 파일 존재 확인
    if [ ! -f "$p" ] || [ ! -f "$i" ] || [ ! -f "$a" ]; then
      echo "  [SKIP] 입력 파일 없음: $in_dir"
      continue
    fi

    # 이미 완료된 init 건너뛰기 (재실행 안전)
    if [ -d "$out_dir/predictions.zarr" ]; then
      echo "  [SKIP] 이미 존재: $out_dir/predictions.zarr"
      continue
    fi

    year_start=$(date +%s)

    printf "%s\n" \
      "2" \
      "$p" \
      "$i" \
      "$a" \
      "$ENSEMBLE" \
      "$DAYS" \
      "$out_dir" \
    | python raw_solar_vmap.py

    rc=$?
    if [ $rc -ne 0 ]; then
      echo "  [ERROR] $year/$mmdd 추론 실패 (rc=$rc)"
    fi

    elapsed=$(( $(date +%s) - year_start ))
    echo "[$(date)] $year/$mmdd 완료 (${elapsed}s = $((elapsed/60))m$((elapsed%60))s)"
  done
done

total=$(( $(date +%s) - start_total ))
echo "[$(date)] 전체 완료 (총 ${total}s = $((total/3600))h$(((total%3600)/60))m)"
