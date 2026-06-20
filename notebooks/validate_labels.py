# %% [markdown]
# # 라벨 검증 — 복합 Hot-Dry (Stage 1)
#
# 30년(1989–2020) 라벨이 **실제 사건과 정성적으로 일치**하는지 확인한다 (게이트 1 조건).
# 입력:
#   - `data/processed/era5_daily_1989-2020.nc`  (정규화된 일평균, ocean=NaN)
#   - `data/labels/labels_1989-2020.nc`         (heatwave/drought/compound int8)
#
# 헤드리스 서버에서도 동작 — `plt.show()` 대신 `notebooks/figures/`에 PNG 저장.
#
# 점검 항목:
# 1. 클래스 균형(전체/월별 양성 비율)
# 2. 복합 사건 공간분포(빈도 지도)
# 3. **2018 동아시아 폭염**(2018-07~08) 대조
# 4. **2016 동아시아 폭염** 대조
# 5. SPI 계절성 점검(월별 가뭄 비율 쏠림)

# %%
import os
from pathlib import Path

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")        # 헤드리스 → 파일 저장 전용
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data/processed/era5_daily_1989-2020.nc"
LABELS = ROOT / "data/labels/labels_1989-2020.nc"
LSM = ROOT / "data/raw/lsm.nc"
FIG = ROOT / "notebooks/figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(name: str):
    """현재 figure를 PNG 저장(헤드리스) + show(인터랙티브)."""
    path = FIG / f"{name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"  [fig] {path}")
    plt.close()


proc = xr.open_dataset(PROC)
lab = xr.open_dataset(LABELS)
land = (xr.open_dataset(LSM)["lsm"] >= 0.5) if LSM.exists() else None
print("proc vars:", list(proc.data_vars), "| time:", proc.sizes.get("time"))
print("label vars:", list(lab.data_vars))

# %% [markdown]
# ## 1. 클래스 균형 (전체 / 월별)

# %%
for v in ["heatwave", "drought", "compound"]:
    if v not in lab:
        continue
    da = lab[v].where(land) if land is not None else lab[v]   # 해양 제외
    frac = float(da.mean())
    pos_w = (1 - frac) / frac if frac > 0 else float("nan")
    print(f"{v:10s} 양성비율={frac:.4f}  pos_weight≈{pos_w:.1f}")

if "compound" in lab:
    cmp = lab["compound"].where(land) if land is not None else lab["compound"]
    by_month = cmp.groupby("time.month").mean(dim=...).values
    plt.figure(figsize=(6, 3))
    plt.bar(range(1, 13), by_month)
    plt.xlabel("month"); plt.ylabel("compound pos frac"); plt.title("월별 복합 사건 비율")
    plt.tight_layout(); save("01_compound_by_month")

# %% [markdown]
# ## 2. 복합 사건 공간분포 (30년 빈도)

# %%
if "compound" in lab:
    freq = (lab["compound"].where(land) if land is not None else lab["compound"]).mean(dim="time")
    plt.figure(figsize=(7, 4))
    freq.plot(x="lon", y="lat", cmap="YlOrRd")
    plt.title("복합 Hot-Dry 빈도 (1989–2020 평균)")
    plt.tight_layout(); save("02_compound_freq_map")

# %% [markdown]
# ## 3·4. 폭염 사건 대조 (2018 / 2016 동아시아)

# %%
def case_study(t0: str, t1: str, title: str, fname: str):
    win_lab = lab.sel(time=slice(t0, t1))
    win_proc = proc.sel(time=slice(t0, t1))
    if win_lab.sizes.get("time", 0) == 0:
        print(f"[{title}] 해당 기간 데이터 없음")
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    t2m = win_proc["t2m"].mean("time")
    (t2m.where(land) if land is not None else t2m).plot(ax=axes[0], x="lon", y="lat", cmap="RdBu_r", center=0)
    axes[0].set_title(f"{title}\nt2m anomaly(norm)")
    if "heatwave" in win_lab:
        hw = win_lab["heatwave"].mean("time")
        (hw.where(land) if land is not None else hw).plot(ax=axes[1], x="lon", y="lat", cmap="YlOrRd", vmin=0, vmax=1)
        axes[1].set_title("heatwave freq")
    if "compound" in win_lab:
        cp = win_lab["compound"].mean("time")
        (cp.where(land) if land is not None else cp).plot(ax=axes[2], x="lon", y="lat", cmap="YlOrRd", vmin=0, vmax=1)
        axes[2].set_title("compound freq")
    plt.tight_layout(); save(fname)
    box = dict(lon=slice(124, 131), lat=slice(33, 39))   # 한반도 박스
    if "heatwave" in win_lab:
        print(f"[{title}] 한반도 heatwave 평균빈도={float(win_lab['heatwave'].sel(**box).mean()):.3f}")
    if "compound" in win_lab:
        print(f"[{title}] 한반도 compound 평균빈도={float(win_lab['compound'].sel(**box).mean()):.3f}")


case_study("2018-07-15", "2018-08-15", "2018 동아시아 폭염", "03_case_2018")
case_study("2016-07-20", "2016-08-25", "2016 동아시아 폭염", "04_case_2016")

# %% [markdown]
# ## 5. SPI 계절성 점검 (월별 가뭄 비율 쏠림)

# %%
if "drought" in lab:
    dr = lab["drought"].where(land) if land is not None else lab["drought"]
    by_month = dr.groupby("time.month").mean(dim=...).values
    plt.figure(figsize=(6, 3))
    plt.bar(range(1, 13), by_month, color="tab:brown")
    plt.xlabel("month"); plt.ylabel("drought pos frac"); plt.title("월별 가뭄 라벨 비율 (쏠림 점검)")
    plt.tight_layout(); save("05_drought_by_month")
    print("월별 가뭄비율 std =", float(np.nanstd(by_month)), "(작을수록 계절중립)")

print("\n검증 그림 저장 완료 → notebooks/figures/")
