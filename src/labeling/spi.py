"""가뭄 라벨링: SPI < threshold (scale_days 누적). 현재 운영=SPI-1(30일)<−1.

(2026-06-03 SPI-3<−1.5 → SPI-1<−1 전환. GenCast 15일 평가 타당성. scale_days는
compound.py가 labeling.yaml의 spi_scale*30으로 전달 → SPI-1=30, SPI-3=90.)

SPI(Standardized Precipitation Index): scale개월 누적강수를 gamma 분포에
적합한 뒤 표준정규로 변환. 0 강수는 혼합분포(P(0) + gamma)로 처리.

**월별 적합(표준 SPI)**: 강수는 계절성이 강하므로(동아시아=여름 폭우/겨울 건조)
누적 끝점의 캘린더 월별로 따로 gamma 분포를 적합한다. 전체를 한 분포로 적합하면
여름 가뭄을 놓치고 겨울을 과탐하는 편향이 생김 → 여름 Hot-Dry 타깃에 치명적.

설계 절충: 표준 SPI는 월해상도(연 12값)에 적합하지만, 우리 탐지 모델은 일 해상도
라벨이 필요해 '일별 90일 누적 → 끝점 월별 적합'을 쓴다(daily-SPI 접근). 같은 달의
일별 윈도우는 자기상관이 크므로 표본 수가 풍부해 보여도 유효 독립표본은 ~연수 수준
임에 유의(30년 전제). 음수 누적강수·degenerate 격자·fit 실패는 격자별로 방어 처리.

주의: SPI-3는 3개월 누적이 필요하므로 1개월 슬라이스로는 계산 불가.
Stage 0에서는 build_drought_mask가 길이 부족을 감지해 더미를 반환한다.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import gamma, norm

_MIN_SAMPLES = 10   # 한 달 표본 최소 개수(30년이면 달당 ~30개)
_MIN_POSITIVE = 5   # gamma 적합에 필요한 양(>0) 강수 최소 개수


def _spi_1d_monthly(precip_accum: np.ndarray, months: np.ndarray) -> np.ndarray:
    """1개 격자 시계열의 누적강수 → SPI 값. 캘린더 월별로 gamma 적합."""
    out = np.full_like(precip_accum, np.nan, dtype=np.float64)
    valid = ~np.isnan(precip_accum)
    acc = np.where(valid, np.clip(precip_accum, 0.0, None), np.nan)  # 음수 누적강수 → 0
    for m in range(1, 13):
        idx = np.where((months == m) & valid)[0]
        if idx.size < _MIN_SAMPLES:
            continue
        x = acc[idx]
        pos = x[x > 0]
        if pos.size < _MIN_POSITIVE or np.ptp(pos) < 1e-8:   # degenerate(분산~0) 격자 skip
            continue
        try:
            a, loc, scale = gamma.fit(pos, floc=0)           # 해당 월 gamma 적합
        except Exception:                                    # fit 실패 격자 보호(벡터화 중단 방지)
            continue
        q = (x.size - pos.size) / x.size             # P(강수=0)
        cdf = np.where(x > 0, q + (1 - q) * gamma.cdf(x, a, loc=loc, scale=scale), q)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        out[idx] = norm.ppf(cdf)
    return out


def compute_spi(tp: xr.DataArray, scale_days: int = 90) -> xr.DataArray:
    """일강수 → scale_days 누적 → 격자별 SPI(월별 적합)."""
    accum = tp.rolling(time=scale_days, min_periods=scale_days).sum()
    months = xr.DataArray(accum["time"].dt.month, dims=["time"],
                          coords={"time": accum["time"]})
    spi = xr.apply_ufunc(
        _spi_1d_monthly, accum, months,
        input_core_dims=[["time"], ["time"]], output_core_dims=[["time"]],
        vectorize=True, dask="parallelized", output_dtypes=[np.float64],
    )
    name = f"spi{max(1, round(scale_days / 30))}"   # scale에 맞는 이름(90일 → spi3)
    return spi.transpose("time", "lat", "lon").rename(name)


def build_drought_mask(tp: xr.DataArray, threshold: float = -1.5,
                       scale_days: int = 90) -> xr.DataArray:
    """SPI(scale_days 누적) < threshold 인 격자·시점을 True. 길이 부족 시 더미(전부 0).

    NaN 처리 주의: SPI가 NaN인 곳(초기 scale_days-1일 누적부족·적합실패·해양)은
    (spi < threshold)에서 False(=비가뭄)로 처리된다. 초기 버퍼 구간은 학습/평가에서
    제외할 것. NaN 비율을 출력해 가시화한다. (전면 NaN 라벨 전파는 dataset/loss
    동시수정이 필요해 라벨 생성 단계에서 다룸.)
    """
    if tp.sizes["time"] < scale_days:
        print(f"[spi] time={tp.sizes['time']} < {scale_days} → SPI({scale_days}일) 계산 불가, 더미 반환")
        dummy = xr.zeros_like(tp, dtype="int8")
        return dummy.rename("drought")
    spi = compute_spi(tp, scale_days)
    nan_frac = float(spi.isnull().mean())
    print(f"[spi] SPI NaN 비율={nan_frac:.3f} (초기 {scale_days - 1}일 누적부족 등 → 비가뭄 처리)")
    return (spi < threshold).astype("int8").rename("drought")
