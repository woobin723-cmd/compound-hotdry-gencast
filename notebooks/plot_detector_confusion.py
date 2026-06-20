"""탐지모델 자체(Pipeline A) confusion matrix — ERA5 Test셋(2015-2020) 입력.

GenCast 입력(Pipeline B·리드별)과 대조: 정답 ERA5 입력 시 탐지모델의 상한 성능.
타깃당 1개 confusion(리드 개념 없음). SPI-1 탐지기·해양 마스킹·calibrate된 thr 사용.
출력: notebooks/figures/detector_confusion_era5.png  +  data/skill_json/detector_era5.json
실행(cecd env): python notebooks/plot_detector_confusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from src.eval.detector import Detector  # noqa: E402
from src.eval.metrics import _counts, all_metrics, reliability  # noqa: E402
from src.train import _broadcast_mask, load_land_mask  # noqa: E402

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

ERA5 = "data/processed/era5_daily_1989-2020.nc"
LABELS = "data/labels/labels_spi1_1989-2020.nc"
MANIFEST = "config/detectors_spi1.json"
TEST = [2015, 2020]
TARGETS = ["drought", "compound", "heatwave"]
KO = {"drought": "가뭄", "compound": "복합", "heatwave": "폭염"}
COL = {"drought": "#2E8B57", "compound": "#C0392B", "heatwave": "#E69500"}
FDIR = ROOT / "figures"
JDIR = ROOT.parent / "data" / "skill_json"


def main():
    mask = load_land_mask(ERA5)
    spec = json.loads(Path(MANIFEST).read_text())   # 타깃별 proc(아노말리 5채널 등) 도출용
    results = {}
    for tgt in TARGETS:
        det = Detector.from_manifest(tgt, path=MANIFEST)
        proc = spec[tgt].get("proc", ERA5)           # 폭염·복합=era5_daily_anom, 가뭄=기본
        prob, t = det.proba(proc, LABELS, TEST)
        tm = _broadcast_mask(mask, prob.shape[0])
        m = all_metrics(prob, t, threshold=det.threshold, mask=tm)
        tp, fp, fn, tn = _counts(prob >= det.threshold, t, tm)
        results[tgt] = {"thr": det.threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "CSI": m["CSI"], "POD": m["Recall_POD"], "Precision": m["Precision"],
                        "AUC": m["AUC_ROC"], "base": m["base_rate"],
                        "reliability": reliability(prob, t, tm)}
        print(f"[{tgt}] thr={det.threshold:.3f} CSI={m['CSI']:.4f} POD={m['Recall_POD']:.3f} "
              f"Prec={m['Precision']:.3f} AUC={m['AUC_ROC']:.3f} TP={tp} FP={fp} FN={fn} TN={tn}")

    JDIR.mkdir(parents=True, exist_ok=True)
    (JDIR / "detector_era5.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # 그림: 1×3 (타깃별 2×2 confusion)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, tgt in zip(axes, TARGETS):
        r = results[tgt]
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])  # 행=실제(+/−), 열=예측(+/−)
        ax.imshow(np.log1p(cm), cmap="Greens", aspect="auto")
        for (yy, xx), v in np.ndenumerate(cm):
            ax.text(xx, yy, f"{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if np.log1p(v) > np.log1p(cm.max()) * 0.6 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["예측+", "예측−"]); ax.set_yticklabels(["실제+", "실제−"])
        ax.set_title(f"{KO[tgt]}  CSI={r['CSI']:.3f} POD={r['POD']:.2f}\n"
                     f"Prec={r['Precision']:.2f} AUC={r['AUC']:.3f}",
                     fontsize=10.5, color=COL[tgt])
    fig.suptitle("탐지모델 Confusion Matrix (Pipeline A)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    FDIR.mkdir(parents=True, exist_ok=True)
    out = FDIR / "detector_confusion_era5.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
