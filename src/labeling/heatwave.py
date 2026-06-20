"""폭염 라벨링: t2m > p95(day-of-year 윈도우) & 3일 이상 연속."""
from __future__ import annotations

import numpy as np
import xarray as xr


def doy_percentile_threshold(t2m: xr.DataArray, percentile: float = 95,
                             window: int = 7) -> xr.DataArray:
    """day-of-year별 ±window 윈도우 백분위 임계 (lat, lon, dayofyear)."""
    doy = t2m["time"].dt.dayofyear
    t2m = t2m.assign_coords(dayofyear=doy)
    thresholds = []
    days = np.arange(1, 367)
    for d in days:
        # 순환 윈도우 (연말/연초 wrap)
        diff = np.abs(((doy - d + 183) % 366) - 183)
        sel = t2m.where(diff <= window, drop=False)
        thresholds.append(sel.quantile(percentile / 100.0, dim="time"))
    thr = xr.concat(thresholds, dim="dayofyear").assign_coords(dayofyear=days)
    return thr.drop_vars("quantile", errors="ignore")


def pooled_percentile_threshold(t2m: xr.DataArray, percentile: float = 90) -> xr.DataArray:
    """격자별 전체기간 풀링 p퍼센타일 (lat, lon). doy 기후값을 만들기엔
    데이터가 짧은 Stage 0 슬라이스용 폴백. (반환 차원: lat, lon)"""
    thr = t2m.quantile(percentile / 100.0, dim="time")
    return thr.drop_vars("quantile", errors="ignore")


def heatwave_mask_pooled(t2m: xr.DataArray, threshold_2d: xr.DataArray,
                         min_consecutive: int = 3) -> xr.DataArray:
    """풀링 임계(lat,lon) 기준 폭염 마스크."""
    exceed = (t2m > threshold_2d).astype("int8")
    return _enforce_consecutive(exceed, min_consecutive).rename("heatwave")


def heatwave_mask(t2m: xr.DataArray, threshold: xr.DataArray,
                  min_consecutive: int = 3) -> xr.DataArray:
    """임계 초과 & 연속 min_consecutive일 이상인 날을 True로."""
    doy = t2m["time"].dt.dayofyear
    thr_t = threshold.sel(dayofyear=doy)            # 각 시점에 해당 doy 임계 매핑
    exceed = (t2m > thr_t).astype("int8")
    mask = _enforce_consecutive(exceed, min_consecutive)
    return mask.rename("heatwave")


def _enforce_consecutive(exceed: xr.DataArray, n: int) -> xr.DataArray:
    """time축에서 연속 n일 이상 True인 구간만 유지 (벡터화)."""
    arr = exceed.values  # (time, lat, lon)
    t = arr.shape[0]
    # 누적 연속 카운트
    run = np.zeros_like(arr, dtype=np.int32)
    run[0] = arr[0]
    for i in range(1, t):
        run[i] = (run[i - 1] + 1) * arr[i]
    # 길이 n 이상 run에 속한 모든 날 마킹 (뒤에서 앞으로 전파)
    keep = run >= n
    for i in range(t - 2, -1, -1):
        keep[i] |= keep[i + 1] & (arr[i] == 1)
    return exceed.copy(data=keep.astype("int8"))
