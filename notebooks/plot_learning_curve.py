"""학습 곡선(Learning Curve) — loss 패널·CSI 패널 분리.

logs/*.log의 `[ep N] train_loss=.. val_CSI=..@thr=.. val_F1=.. val_AUC=..`와
`[best-val] CSI=.. (epN)` 줄을 파싱.
**복합 4멤버 앙상블(전부 Focal·통제변수=모델 구조)** 학습곡선. 손실 통일이라
좌 패널 train_loss 절대비교 가능(동일 손실·동일 타깃·동일 분할/오버샘플).
우 패널=val_CSI(★=조기종료 best-val).
출력: notebooks/figures/learning_curves.png
실행(cecd env): python notebooks/plot_learning_curve.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
LOGS = ROOT.parent / "logs"
FDIR = ROOT / "figures"

EP_RE = re.compile(r"\[ep\s*(\d+)\]\s*train_loss=([\d.]+)\s*val_CSI=([\d.]+).*?val_AUC=([\d.]+)")
BEST_RE = re.compile(r"\[best-val\]\s*CSI=([\d.]+).*?\(ep(\d+)\)")

# 복합 4멤버 앙상블 (라벨, 로그파일, 색, 손실명) — 전부 Focal·구조만 통제변수
MODELS = [
    ("CNN-LSTM (Focal)", "cnn_lstm_compound_spi1_focal.log", "#C0392B", "Focal"),
    ("ConvLSTM (Focal)", "convlstm_compound_spi1_focal.log", "#8E44AD", "Focal"),
    ("U-Net (Focal)", "unet_compound_spi1_focal.log", "#16A085", "Focal"),
    ("Transformer (Focal)", "transformer_compound_spi1.log", "#E69500", "Focal"),
]


def parse(path: Path):
    eps, loss, csi, auc = [], [], [], []
    best = None
    for ln in path.read_text().splitlines():
        m = EP_RE.search(ln)
        if m:
            eps.append(int(m.group(1))); loss.append(float(m.group(2)))
            csi.append(float(m.group(3))); auc.append(float(m.group(4)))
        b = BEST_RE.search(ln)
        if b:
            best = (int(b.group(2)), float(b.group(1)))
    return eps, loss, csi, auc, best


def main():
    fig, (axL, axC) = plt.subplots(1, 2, figsize=(14, 6))

    for label, fname, col, lname in MODELS:
        p = LOGS / fname
        if not p.exists():
            continue
        eps, loss, csi, auc, best = parse(p)

        # 좌: train_loss (log축)
        axL.plot(eps, loss, "-o", color=col, lw=2, ms=5, label=label)

        # 우: val_CSI + best epoch 마커
        axC.plot(eps, csi, "-s", color=col, lw=2, ms=5, label=label)
        if best:
            be, bc = best
            axC.plot(be, bc, "*", color=col, ms=16, mec="black", mew=0.8, zorder=5)

    axL.set_yscale("log")
    axL.set_xlabel("epoch"); axL.set_ylabel("train_loss (log scale)")
    axL.set_title("① train_loss", fontsize=12)
    axL.grid(alpha=0.3, which="both"); axL.legend(fontsize=9)

    axC.set_xlabel("epoch"); axC.set_ylabel("val_CSI")
    axC.set_ylim(0, 0.25)
    axC.set_title("② val_CSI (★=best-val)", fontsize=12)
    axC.grid(alpha=0.3); axC.legend(fontsize=9, loc="lower right")

    fig.suptitle("복합 4멤버 학습 곡선", fontsize=14, fontweight="bold")
    fig.tight_layout()
    FDIR.mkdir(parents=True, exist_ok=True)
    out = FDIR / "learning_curves.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
