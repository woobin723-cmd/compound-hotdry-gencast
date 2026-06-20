"""앙상블 멤버 ablation 시각화 (슬라이드 7-3용).

  fig1 ablation_csi.png   — 멤버셋 조합별 Test CSI 가로 바(단독→4멤버, 최적 강조)
  fig2 member_corr.png    — 멤버쌍 예측 확률상관 히트맵(낮을수록 다양성↑)

입력: data/skill_json/ablation_compound_spi1.json (src.eval.ablation 산출)
출력: notebooks/figures/{ablation_csi,member_corr}.png
실행(cecd env): python notebooks/plot_ablation.py
"""
from __future__ import annotations

import json
from itertools import combinations
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
SIZE_COL = {1: "#B0B0B0", 2: "#7FB3D5", 3: "#5499C7", 4: "#C0392B"}  # 4멤버=강조


def fig_csi(d):
    rows = sorted(d["rows"], key=lambda x: x["test_CSI"])  # 아래=낮음, 위=높음
    labels = [r["members"] for r in rows]
    csi = [r["test_CSI"] for r in rows]
    cols = [SIZE_COL[r["size"]] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 8))
    y = np.arange(len(rows))
    ax.barh(y, csi, color=cols, edgecolor="black", lw=0.5)
    best = max(rows, key=lambda x: x["test_CSI"])["members"]
    for i, r in enumerate(rows):
        ax.text(r["test_CSI"] + 0.002, i, f"{r['test_CSI']:.3f}", va="center", fontsize=8.5,
                fontweight="bold" if r["members"] == best else "normal")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, fontfamily="monospace")
    for tick, r in zip(ax.get_yticklabels(), rows):   # 채택 멤버셋 라벨 강조
        if r["members"] == best:
            tick.set_fontweight("bold"); tick.set_color("#C0392B")
    ax.set_xlabel("Test CSI (복합·base 0.81%)")
    ax.set_xlim(0, max(csi) * 1.15)
    leg = [plt.Rectangle((0, 0), 1, 1, fc=SIZE_COL[s], ec="black") for s in (1, 2, 3, 4)]
    ax.legend(leg, ["단독", "2멤버", "3멤버", "4멤버(채택)"], loc="lower right", fontsize=9,
              title="조합 크기")
    ax.set_title("멤버셋 조합별 Test CSI", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = FDIR / "ablation_csi.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def fig_corr(d):
    keys = list(d["legend"])             # [C, V, U, T]
    n = len(keys)
    pc = d["pair_corr"]
    M = np.full((n, n), np.nan)          # 대각선=NaN(자기상관 1.0은 자명·마스킹)
    for (a, b) in combinations(keys, 2):
        c = pc.get(f"{a}-{b}", pc.get(f"{b}-{a}"))   # 키 방향 무관 가드
        if c is None:
            raise KeyError(f"pair_corr에 {a}-{b}/{b}-{a} 없음")
        i, j = keys.index(a), keys.index(b)
        M[i, j] = M[j, i] = c
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    cmap = matplotlib.colormaps["RdYlGn_r"].copy(); cmap.set_bad("#EAEAEA")  # 대각선 회색
    im = ax.imshow(M, cmap=cmap, vmin=0.5, vmax=0.7)  # off-diag 0.53~0.68 강조·낮을수록 초록
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    names = [d["legend"][k] for k in keys]
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    lo = np.nanmin(M)
    for i in range(n):
        for j in range(n):
            txt = "—" if i == j else f"{M[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="black", fontweight="bold" if i != j and M[i, j] <= lo + 1e-9 else "normal")
    ax.set_title("멤버쌍 예측 확률상관", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.85, label="확률 상관")
    fig.tight_layout()
    out = FDIR / "member_corr.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def main():
    d = json.loads((JDIR / "ablation_compound_spi1.json").read_text())
    FDIR.mkdir(parents=True, exist_ok=True)
    fig_csi(d)
    fig_corr(d)


if __name__ == "__main__":
    main()
