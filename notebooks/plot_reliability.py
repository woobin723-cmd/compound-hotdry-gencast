"""Reliability diagram (신뢰도 다이어그램) — ERA5 탐지모델 vs GenCast 두 패널.

확률 예보 보정 평가: x=평균 예측확률, y=실제 관측빈도. 대각선=완벽 보정,
아래로 처지면 과신(over-confident). 좌=Pipeline A(ERA5 Test 입력·탐지모델 상한),
우=Pipeline B(GenCast 하이브리드·전체 리드 합산). 3타깃 곡선.
입력: data/skill_json/detector_era5.json · leadtime_{target}.json
출력: notebooks/figures/reliability_diagram.png
실행(cecd env): python notebooks/plot_reliability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
JDIR = ROOT.parent / "data" / "skill_json"
FDIR = ROOT / "figures"
TARGETS = ["drought", "compound", "heatwave"]
KO = {"drought": "가뭄", "compound": "복합", "heatwave": "폭염"}
COL = {"drought": "#2E8B57", "compound": "#C0392B", "heatwave": "#E69500"}


def curve(ax, rel, color, label):
    """bin별 (mean_pred, obs_freq) 곡선. nan/빈 bin 제외."""
    mp = np.array(rel["mean_pred"], dtype=float)
    of = np.array(rel["obs_freq"], dtype=float)
    ok = ~(np.isnan(mp) | np.isnan(of))
    ax.plot(mp[ok], of[ok], "-o", color=color, lw=2, ms=5, label=label)


def panel(ax, get_rel, title):
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="완벽 보정")
    for tgt in TARGETS:
        rel = get_rel(tgt)
        if rel is not None:
            curve(ax, rel, COL[tgt], KO[tgt])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("평균 예측확률"); ax.set_ylabel("실제 관측빈도")
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    ax.set_aspect("equal")


def main():
    era5 = json.loads((JDIR / "detector_era5.json").read_text())
    lead = {t: json.loads((JDIR / f"leadtime_{t}.json").read_text()) for t in TARGETS}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panel(axes[0], lambda t: era5.get(t, {}).get("reliability"),
          "Pipeline A (탐지모델)")
    panel(axes[1], lambda t: lead.get(t, {}).get("reliability_gencast"),
          "Pipeline B (GenCast)")
    fig.suptitle("Reliability Diagram",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    FDIR.mkdir(parents=True, exist_ok=True)
    out = FDIR / "reliability_diagram.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
