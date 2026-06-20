"""리드타임 Skill Decay 시각화 (PPT용).

skill_decay.py --save-json 출력(타깃별 JSON)을 읽어 그림 생성:
  fig1 skill_decay_curves.png  — 리드별 AUC·상대SS(3타깃, GenCast vs ERA5 perfect/clim)
  fig2 confusion_matrices.png  — 타깃별 대표 리드(Day1/5/10/15) confusion matrix
  fig3 spread_error.png        — 리드별 앙상블 spread vs RMSE
실행(cecd env): python notebooks/plot_skill_decay.py
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
FDIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["drought", "compound", "heatwave"]
KO = {"drought": "가뭄", "compound": "복합", "heatwave": "폭염"}
COL = {"drought": "#2E8B57", "compound": "#C0392B", "heatwave": "#E69500"}


def load(target):
    p = JDIR / f"leadtime_{target}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_curves(data):
    """리드별 AUC(좌)·상대SS(우). 3타깃 GenCast 실선, perfect 점선, clim 회색."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for tgt in TARGETS:
        d = data.get(tgt)
        if d is None:
            continue
        leads = d["leads"]; c = COL[tgt]
        axes[0].plot(leads, d["gencast"]["AUC"], "-o", color=c, lw=2, ms=4, label=f"{KO[tgt]} GenCast")
        axes[0].plot(leads, d["perfect"]["AUC"], "--", color=c, lw=1.2, alpha=0.6)
        axes[1].plot(leads, d["rel_ss_csi"], "-o", color=c, lw=2, ms=4, label=KO[tgt])
    axes[0].set_title("리드별 AUC", fontsize=12)
    axes[0].set_xlabel("리드 (Day)"); axes[0].set_ylabel("AUC-ROC")
    axes[0].axhline(0.5, color="gray", ls=":", lw=1); axes[0].set_ylim(0.45, 1.0)
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9, loc="lower left")
    # 주지표=CSI 기반 상대SS(clim CSI=0이라 분모 안정·양수). AUC 기반은 clim AUC 0.9 강세로
    # 후반 음수가 되어 발표용으로 부적합 → 본 그림은 CSI 기반만(8-13 텍스트에 AUC판 음수 해설).
    axes[1].plot([], [], " ", label="가뭄≫복합≈폭염")
    axes[1].set_title("상대 Skill Score (CSI)", fontsize=12)
    axes[1].set_xlabel("리드 (Day)"); axes[1].set_ylabel("상대 SS (CSI)")
    axes[1].axhline(0.5, color="gray", ls=":", lw=1); axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=10)
    fig.suptitle("GenCast 리드타임 Skill Decay", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FDIR / "skill_decay_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def fig_confusion(data):
    """타깃(행)×대표 리드(열) confusion matrix 2×2 히트맵."""
    show_leads = [1, 5, 10, 15]
    fig, axes = plt.subplots(len(TARGETS), len(show_leads), figsize=(13, 9.5))
    for r, tgt in enumerate(TARGETS):
        d = data.get(tgt)
        for cidx, L in enumerate(show_leads):
            ax = axes[r, cidx]
            if d is None:
                ax.axis("off"); continue
            i = d["leads"].index(L)
            cm = np.array([[d["confusion"]["TP"][i], d["confusion"]["FN"][i]],
                           [d["confusion"]["FP"][i], d["confusion"]["TN"][i]]])
            # 행=예측(양성/음성)? 표준: 행=실제, 열=예측. 여기선 [[TP,FN],[FP,TN]] = 행 실제(양/음)·열 예측(양/음)
            ax.imshow(np.log1p(cm), cmap="Blues", aspect="auto")
            for (yy, xx), v in np.ndenumerate(cm):
                ax.text(xx, yy, f"{v:,}", ha="center", va="center", fontsize=10,
                        color="white" if np.log1p(v) > np.log1p(cm.max()) * 0.6 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["예측+", "예측−"], fontsize=8)
            ax.set_yticklabels(["실제+", "실제−"], fontsize=8)
            pod = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) else 0
            ax.set_title(f"{KO[tgt]} Day{L}  (POD={pod:.2f})", fontsize=10, color=COL[tgt])
    fig.suptitle("Confusion Matrix (타깃 × 리드)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FDIR / "confusion_matrices.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def fig_spread(data):
    """리드별 앙상블 spread vs RMSE(error). 잘 보정된 앙상블은 둘이 근접."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(14, 4.5), sharex=True)
    for ax, tgt in zip(axes, TARGETS):
        d = data.get(tgt)
        if d is None:
            ax.axis("off"); continue
        leads = d["leads"]
        ax.plot(leads, d["error"], "-o", color=COL[tgt], lw=2, ms=4, label="RMSE(error)")
        ax.plot(leads, d["spread"], "--s", color="gray", lw=1.6, ms=3, label="앙상블 spread")
        ax.set_title(f"{KO[tgt]}", fontsize=12, color=COL[tgt])
        ax.set_xlabel("리드 (Day)"); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel("확률 단위")
    fig.suptitle("Spread-Error (spread vs RMSE)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FDIR / "spread_error.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


def main():
    data = {t: load(t) for t in TARGETS}
    if all(v is None for v in data.values()):
        raise FileNotFoundError(f"{JDIR}/leadtime_*.json 없음 — skill_decay --save-json 먼저 실행")
    fig_curves(data)
    fig_confusion(data)
    fig_spread(data)


if __name__ == "__main__":
    main()
