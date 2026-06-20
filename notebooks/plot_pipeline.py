"""전체 파이프라인 구조 그림 (PPT용).

Pipeline A(탐지 모델 학습) + Pipeline B(GenCast 예보 평가), Frozen 탐지기 가중치 공유.
출력: notebooks/figures/pipeline_overview.png
실행(cecd env): python notebooks/plot_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 한글 폰트(Noto Sans CJK = 한중일 통합 글리프)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# 색상 팔레트
C_A = "#2E6DB4"      # Pipeline A (파랑)
C_A_FILL = "#DCE9F7"
C_B = "#2E8B57"      # Pipeline B (초록)
C_B_FILL = "#DCF1E4"
C_SHARE = "#C0392B"  # 공유 가중치 강조(빨강)
C_OUT = "#E69500"    # 산출/평가(주황)


def box(ax, x, y, w, h, lines, edge, fill, fontsize=10.5, bold_first=True):
    """둥근 박스 + 멀티라인 텍스트."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2))
    if isinstance(lines, str):
        lines = [lines]
    cx, cy = x + w / 2, y + h / 2
    n = len(lines)
    for i, ln in enumerate(lines):
        yy = cy + (n - 1) * 0.105 / 2 - i * 0.105
        weight = "bold" if (i == 0 and bold_first) else "normal"
        fs = fontsize if (i == 0 and bold_first) else fontsize - 1.3
        col = edge if (i == 0 and bold_first) else "#222222"
        ax.text(cx, yy, ln, ha="center", va="center", fontsize=fs,
                fontweight=weight, color=col, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#555555", style="-|>", ls="-", lw=1.8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=16,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    ax.text(8, 8.6, "전체 파이프라인 구조",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1a1a1a")
    ax.text(8, 8.18, "동아시아 60–180°E, 0–70°N · 1° · 1989–2020 · 변수 t2m·tp·Z500·t850",
            ha="center", va="center", fontsize=10.5, color="#666666")

    # ───────── Pipeline A (학습) — 상단 행 ─────────
    ax.text(0.3, 7.5, "Pipeline A · 탐지 모델 학습 (ERA5)", ha="left", va="center",
            fontsize=12.5, fontweight="bold", color=C_A)
    yA, hA, wA = 6.0, 1.15, 2.55
    xs = [0.3, 3.15, 6.0, 8.85, 11.7]
    boxesA = [
        ["WeatherBench2", "ERA5 zarr", "6h · 1989–2020"],
        ["전처리", "일집계(tp=합)", "Z500=z/g · z-score"],
        ["라벨링", "폭염 t2m>p95&3일", "가뭄 SPI-1<−1 · 복합"],
        ["탐지모델 학습", "4멤버 앙상블", "30일窗 · Focal 통일"],
        ["Frozen 탐지기", "가중치 *.pt", "복합/가뭄/폭염"],
    ]
    edgesA = [C_A, C_A, C_A, C_A, C_SHARE]
    fillsA = [C_A_FILL, C_A_FILL, C_A_FILL, C_A_FILL, "#F7DCDC"]
    for x, b, e, f in zip(xs, boxesA, edgesA, fillsA):
        box(ax, x, yA, wA, hA, b, e, f)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + wA, yA + hA / 2, xs[i + 1], yA + hA / 2, color=C_A)

    # ───────── Pipeline B (평가) — 하단 행 ─────────
    ax.text(0.3, 4.3, "Pipeline B · GenCast 예보 평가 (Frozen 탐지기 공유)", ha="left",
            va="center", fontsize=12.5, fontweight="bold", color=C_B)
    yB, hB, wB = 2.7, 1.15, 2.35
    xsB = [0.3, 2.95, 5.6, 8.25, 10.9, 13.55]
    boxesB = [
        ["ERA5 초기조건", "여름 90 init", "6/15~8/24·5일"],
        ["GenCast 추론", "8멤버 × 15일", "Frozen·12h"],
        ["후처리", "일집계·동아시아", "Train통계 재정규화"],
        ["하이브리드 윈도우", "ERA5 과거+GenCast", "= 44일(리드1~15)"],
        ["탐지기(Frozen)", "A에서 공유", "확률맵 산출"],
        ["Skill Decay", "리드 Day1~15", "vs perfect/clim"],
    ]
    for i, (x, b) in enumerate(zip(xsB, boxesB)):
        e = C_SHARE if i == 4 else (C_OUT if i == 5 else C_B)
        f = "#F7DCDC" if i == 4 else ("#FBEBCF" if i == 5 else C_B_FILL)
        box(ax, x, yB, wB, hB, b, e, f)
    for i in range(len(xsB) - 1):
        arrow(ax, xsB[i] + wB, yB + hB / 2, xsB[i + 1], yB + hB / 2, color=C_B)

    # ───────── 가중치 공유 화살표 (A Frozen → B 탐지기) ─────────
    x_share_a = xs[4] + wA / 2          # A Frozen 박스 중앙
    x_share_b = xsB[4] + wB / 2         # B 탐지기 박스 중앙
    arrow(ax, x_share_a, yA, x_share_b, yB + hB, color=C_SHARE, style="-|>", ls=(0, (4, 2)), lw=2.2)
    ax.text((x_share_a + x_share_b) / 2 + 1.15, (yA + yB + hB) / 2,
            "가중치 공유\n(재학습 없음)", ha="center", va="center", fontsize=9.5,
            color=C_SHARE, fontweight="bold")

    # ───────── 하단 결과 요약 배너 ─────────
    ax.add_patch(FancyBboxPatch((0.3, 0.45), 15.4, 1.25, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.5, edgecolor="#999999", facecolor="#F4F4F4", zorder=2))
    ax.text(0.6, 1.35, "핵심 결과 (6년 90 init 통합, AUC: Day1→Day15)", ha="left", va="center",
            fontsize=11, fontweight="bold", color="#333333")
    ax.text(0.6, 0.85,
            "가뭄 0.956→0.840(지속성·2주 유효)   복합 0.964→0.661   폭염 0.924→0.602(~1주 한계)   "
            "│ 복합 한계는 폭염 성분이 주요 병목 · ERA5 perfect 평탄(상한선) · 복합=4멤버 앙상블",
            ha="left", va="center", fontsize=10, color="#333333")

    out = Path(__file__).resolve().parent / "figures" / "pipeline_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
