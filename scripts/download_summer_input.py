#!/usr/bin/env python3
"""
GenCast mini 여름철 추론을 위한 ERA5 입력 데이터 다운로드.

Init 날짜: 6/15~8/24 (5일 간격, 15개/년) × 2015~2020 = 90 init
각 init에 대해 D-2, D-1 (00Z + 12Z) = 4 timestep 저장.

전략: (year, month) 단위 bulk 다운로드 → init별 추출
      → CDS 요청 수: 6년 × 3개월 × 3종 = 54개 (vs per-init 270개)

출력 구조:
  /data/woobin/deeplearning/data/gencast_input/{year}/{MMDD}/
    ├── pressure_level.nc   (valid_time=4, pressure_level=13, lat=181, lon=360)
    ├── single_instant.nc   (valid_time=4, lat=181, lon=360)
    └── single_accum.nc     (valid_time=4, lat=181, lon=360)
"""

import cdsapi
import xarray as xr
import numpy as np
import os
from datetime import date, timedelta
import tempfile

SAVE_DIR = "/data/woobin/deeplearning/data/gencast_input"
TMP_DIR = "/data/woobin/tmp"
YEARS = list(range(2015, 2021))

# 6~8월, 5일 간격 15개 init
INIT_DATES = [
    (6, 15), (6, 20), (6, 25), (6, 30),
    (7,  5), (7, 10), (7, 15), (7, 20), (7, 25), (7, 30),
    (8,  4), (8,  9), (8, 14), (8, 19), (8, 24),
]

PRESSURE_LEVELS = [
    "50", "100", "150", "200", "250", "300",
    "400", "500", "600", "700", "850", "925", "1000",
]

client = cdsapi.Client()


# ── 헬퍼 함수 ──────────────────────────────────────────────────────────────

def get_inits_for_month(year, month):
    """해당 (year, month)의 모든 init에 대해 (mmdd, d2_date, d1_date) 반환."""
    result = []
    for (m, d) in INIT_DATES:
        if m != month:
            continue
        init = date(year, m, d)
        d1 = init - timedelta(days=1)
        d2 = init - timedelta(days=2)
        # 모든 init에서 D-2/D-1은 같은 달에 속함 (설계상 보장)
        assert d1.month == d2.month == month, (
            f"init={year}/{m:02d}/{d:02d}의 D-2({d2})/D-1({d1})이 다른 달 — "
            "현재 설계는 D-2/D-1이 같은 달에 있음을 가정함"
        )
        result.append((f"{m:02d}{d:02d}", d2, d1))
    return result


def needed_days(year, month):
    """bulk 다운로드에 필요한 날짜 목록 (zero-padded string 리스트, 정렬)."""
    days = set()
    for (_, d2, d1) in get_inits_for_month(year, month):
        days.add(f"{d2.day:02d}")
        days.add(f"{d1.day:02d}")
    return sorted(days)


def all_done(year, month):
    for (mmdd, _, _) in get_inits_for_month(year, month):
        out_dir = os.path.join(SAVE_DIR, str(year), mmdd)
        for fname in ["pressure_level.nc", "single_instant.nc", "single_accum.nc"]:
            if not os.path.exists(os.path.join(out_dir, fname)):
                return False
    return True


def extract_and_save(ds_bulk, year, month, fname):
    """bulk 데이터셋에서 init별 4-timestep slice 추출 후 저장."""
    ds_bulk = ds_bulk.sortby("valid_time")
    for (mmdd, d2, d1) in get_inits_for_month(year, month):
        out_path = os.path.join(SAVE_DIR, str(year), mmdd, fname)
        if os.path.exists(out_path):
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 4 timestep: D-2 00Z, D-2 12Z, D-1 00Z, D-1 12Z
        times = np.array([
            f"{d2.year}-{d2.month:02d}-{d2.day:02d}T00:00",
            f"{d2.year}-{d2.month:02d}-{d2.day:02d}T12:00",
            f"{d1.year}-{d1.month:02d}-{d1.day:02d}T00:00",
            f"{d1.year}-{d1.month:02d}-{d1.day:02d}T12:00",
        ], dtype="datetime64[ns]")
        ds_slice = ds_bulk.sel(valid_time=times)
        ds_slice.load()
        ds_slice.to_netcdf(out_path)
        print(f"    [saved] {year}/{mmdd}/{fname}")


# ── 메인 다운로드 루프 ──────────────────────────────────────────────────────

os.makedirs(TMP_DIR, exist_ok=True)

total_months = len(YEARS) * 3
done_months = 0

for year in YEARS:
    for month in [6, 7, 8]:
        done_months += 1
        if all_done(year, month):
            print(f"[SKIP] {year}/{month:02d}  ({done_months}/{total_months})")
            continue

        days = needed_days(year, month)
        print(f"\n[{done_months}/{total_months}] {year}/{month:02d}  days={days}")

        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmpdir:

            # ── 1. Pressure level ──────────────────────────────────────────
            p_tmp = os.path.join(tmpdir, "pressure.nc")
            print(f"  Downloading pressure level...")
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "geopotential", "specific_humidity", "temperature",
                        "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
                    ],
                    "pressure_level": PRESSURE_LEVELS,
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days,
                    "time": ["00:00", "12:00"],
                    "grid": [1.0, 1.0],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
            ).download(p_tmp)
            ds_p = xr.open_dataset(p_tmp, engine="netcdf4")
            extract_and_save(ds_p, year, month, "pressure_level.nc")
            ds_p.close()

            # ── 2. Single instant ──────────────────────────────────────────
            i_tmp = os.path.join(tmpdir, "instant.nc")
            print(f"  Downloading single instant...")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": [
                        "10m_u_component_of_wind", "10m_v_component_of_wind",
                        "2m_temperature", "mean_sea_level_pressure",
                        "sea_surface_temperature", "geopotential", "land_sea_mask",
                    ],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days,
                    "time": ["00:00", "12:00"],
                    "grid": [1.0, 1.0],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
            ).download(i_tmp)
            ds_i = xr.open_dataset(i_tmp, engine="netcdf4")
            extract_and_save(ds_i, year, month, "single_instant.nc")
            ds_i.close()

            # ── 3. Single accum (total precipitation) ─────────────────────
            a_tmp = os.path.join(tmpdir, "accum.nc")
            print(f"  Downloading single accum...")
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": ["total_precipitation"],
                    "year": str(year),
                    "month": f"{month:02d}",
                    "day": days,
                    "time": ["00:00", "12:00"],
                    "grid": [1.0, 1.0],
                    "data_format": "netcdf",
                    "download_format": "unarchived",
                },
            ).download(a_tmp)
            ds_a = xr.open_dataset(a_tmp, engine="netcdf4")
            extract_and_save(ds_a, year, month, "single_accum.nc")
            ds_a.close()

        print(f"  [DONE] {year}/{month:02d}")

print("\n=== 전체 다운로드 완료 ===")
print(f"저장 경로: {SAVE_DIR}")
