"""복합 Hot-Dry 라벨: 동일 격자·동일 시기 폭염 & 가뭄 동시 충족.

실행(data/cecd env):
  python -m src.labeling.compound --tag 2015-07
  python -m src.labeling.compound --tag 1989-2020
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.labeling.heatwave import (  # noqa: E402
    doy_percentile_threshold,
    heatwave_mask,
    heatwave_mask_pooled,
    pooled_percentile_threshold,
)
from src.labeling.spi import build_drought_mask  # noqa: E402
from src.utils import load_config, resolve_path  # noqa: E402


def build_labels(daily: xr.Dataset, lab_cfg: dict) -> xr.Dataset:
    """일평균 데이터셋(t2m, tp, ...) → heatwave/drought/compound 라벨셋.

    주의: daily는 정규화 *이전*의 물리값이어야 임계가 의미를 가진다.
    """
    hw = lab_cfg["heatwave"]
    # doy 기후값은 다년 데이터 필요. 1년 미만 슬라이스(Stage 0)는 풀링 폴백.
    if daily.sizes["time"] >= 365:
        thr = doy_percentile_threshold(daily["t2m"], hw["percentile"], hw["doy_window"])
        heat = heatwave_mask(daily["t2m"], thr, hw["min_consecutive_days"])
    else:
        print(f"[heatwave] time={daily.sizes['time']} < 365 → 풀링 p{90} 폴백(Stage 0 smoke)")
        thr = pooled_percentile_threshold(daily["t2m"], 90)
        heat = heatwave_mask_pooled(daily["t2m"], thr, hw["min_consecutive_days"])

    dr = lab_cfg["drought"]
    drought = build_drought_mask(daily["tp"], dr["threshold"], dr["spi_scale"] * 30)

    drought = drought.reindex_like(heat, fill_value=0)
    compound = (heat.astype("int8") & drought.astype("int8")).astype("int8").rename("compound")
    return xr.Dataset({"heatwave": heat, "drought": drought, "compound": compound})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/data.yaml")
    ap.add_argument("--labeling", default="config/labeling.yaml")
    ap.add_argument("--tag", required=True, help="예: 2015-07 또는 1989-2020")
    ap.add_argument("--raw-daily", default=None,
                    help="물리값 일평균 nc 경로(정규화 전). 없으면 processed에서 역추정 불가 → 직접 지정")
    ap.add_argument("--out", default=None,
                    help="출력 nc 경로 직접 지정(기본 labels_{tag}.nc). SPI-1 등 변형 라벨을 "
                         "기존 SPI-3 산출물 덮어쓰지 않고 별도 저장할 때 사용.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    lab_cfg = load_config(args.labeling)
    lab_dir = resolve_path(cfg["paths"]["labels"])
    lab_dir.mkdir(parents=True, exist_ok=True)

    # 라벨은 물리값 기준. preprocess가 정규화 전 일평균도 저장하도록 하거나,
    # 여기서는 raw 6h에서 직접 일집계한 물리값을 받는다.
    from src.data.preprocess import daily_aggregate, load_raw

    if args.tag.count("-") == 1 and len(args.tag) == 7:   # YYYY-MM
        labels = [args.tag]
    else:
        s, e = args.tag.split("-")
        labels = [str(y) for y in range(int(s), int(e) + 1)]
    raw_dir = resolve_path(cfg["paths"]["raw"])
    daily = daily_aggregate(load_raw(raw_dir, labels).load())

    labelset = build_labels(daily, lab_cfg)
    out = Path(args.out) if args.out else lab_dir / f"labels_{args.tag}.nc"
    out.parent.mkdir(parents=True, exist_ok=True)
    labelset.to_netcdf(out)
    for v in labelset.data_vars:
        frac = float(labelset[v].mean())
        print(f"[labels] {v}: positive_frac={frac:.4f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
