"""앙상블 멤버 다양성 매트릭스 개념도 (슬라이드 6·7-3용).

4멤버를 **시간 처리 방식(y) × 공간 처리 방식(x)** 평면에 배치해 "서로 다른 칸을
점유한다(구조적 상보성)"는 설계 의도를 직관화. 위치는 설계 의도 기반 정성 좌표
(정량 근거는 member_corr·ablation_csi). 마커 크기=파라미터 수, 주석=단독 Test CSI.

출력: notebooks/figures/diversity_matrix.png
실행(cecd env): python notebooks/plot_diversity.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

FDIR = Path(__file__).resolve().parent / "figures"

# 멤버: (이름, 색, 공간처리 x, 시간처리 y, 파라미터M, 단독 Test CSI, 시간설명, 공간설명)
MEMBERS = [
    ("CNN-LSTM",   "#C0392B", 0.15, 0.85, 2.49, 0.146, "격자별 순환",   "평탄화(격자독립)"),
    ("ConvLSTM",   "#8E44AD", 0.55, 0.80, 0.49, 0.130, "conv 게이트 순환", "국소 conv(공간보존)"),
    ("Transformer","#E69500", 0.68, 0.55, 6.41, 0.129, "시간 어텐션",   "패치 토큰 어텐션"),
    ("U-Net",      "#16A085", 0.90, 0.18, 1.96, 0.174, "채널 concat",   "encoder-decoder+skip"),
]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 8))
    for name, col, x, y, pm, csi, ts, ss in MEMBERS:
        ax.scatter(x, y, s=pm * 260 + 320, color=col, alpha=0.82, edgecolor="black",
                   lw=1.5, zorder=3)
        ax.annotate(f"시간: {ts}\n공간: {ss}", (x, y), textcoords="offset points",
                    xytext=(0, 46), ha="center", fontsize=8.4, color=col, fontweight="bold")
        ax.annotate(f"{name}\n{pm:.2f}M · CSI {csi:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -46), ha="center", va="center", fontsize=9.5,
                    color=col, fontweight="bold", zorder=4)

    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("공간 처리 — 격자독립 ───────────▶ 강한 공간 prior", fontsize=11)
    ax.set_ylabel("시간 처리 — 채널화 ───────────▶ 순환·어텐션", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(alpha=0.25)
    # 사분면 가이드
    ax.axhline(0.5, color="gray", ls=":", lw=0.8); ax.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("앙상블 멤버 다양성 매트릭스 (개념도)", fontsize=13)
    fig.tight_layout()
    FDIR.mkdir(parents=True, exist_ok=True)
    out = FDIR / "diversity_matrix.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
