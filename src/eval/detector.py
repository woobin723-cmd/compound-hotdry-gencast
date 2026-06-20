"""탐지 모델 추론 진입점 (Pipeline A→B 핸드오프).

학습된 체크포인트는 자급자족(가중치+config+윈도우+val_threshold)이라 재학습 없이 즉시 추론 가능.
이 모듈은 그 위에 **앙상블 결합규칙·고정 임계값**을 매니페스트로 고정해, 다음 세션·Pipeline B에서
"로드→추론"만으로 이어지게 한다.

매니페스트: config/detectors.json
  - 단일 모델(가뭄·폭염): threshold=null → 멤버 ckpt의 val_threshold 사용.
  - 앙상블(복합): combine="mean", threshold=고정값(calibrate로 Val에서 1회 산출).

사용:
  # 앙상블 임계값 보정 후 매니페스트 기록(최초 1회)
  python -m src.eval.detector --target compound --calibrate \
      --proc data/processed/era5_daily_1989-2020.nc --labels data/labels/labels_1989-2020.nc
  # 추론·평가(라벨 있으면 지표까지)
  python -m src.eval.detector --target compound --split test \
      --proc data/processed/era5_daily_1989-2020.nc --labels data/labels/labels_1989-2020.nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import all_metrics, best_threshold  # noqa: E402
from src.train import (build_model, collect_probs, load_land_mask,  # noqa: E402
                       make_loader, _broadcast_mask)
from src.utils import load_config  # noqa: E402

MANIFEST = "config/detectors.json"


def _load_member(ckpt_path, device):
    """체크포인트 → (로드된 모델, cfg, val_threshold, target)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, float(ckpt.get("val_threshold", 0.5)), ckpt.get("target")


class Detector:
    """단일 또는 앙상블 탐지기. 멤버별 윈도우(멀티스케일 포함)는 각 ckpt config에서 복원."""

    def __init__(self, target, members, combine, threshold, device=None):
        self.target = target
        self.combine = combine            # "single" | "mean"
        self.threshold = threshold        # 고정 임계값(앙상블) 또는 None(단일→멤버 val_threshold)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = [_load_member(m, self.device) for m in members]
        if combine == "single" and self.threshold is None:
            self.threshold = self._loaded[0][2]   # 단일: 멤버 val_threshold 채택

    @classmethod
    def from_manifest(cls, target, path=MANIFEST, device=None):
        spec = json.loads(Path(path).read_text())[target]
        return cls(target, spec["members"], spec["combine"], spec.get("threshold"), device)

    def proba(self, proc, labels, years):
        """멤버 확률 평균(앙상블) 또는 단일 확률. (prob, target_label) 반환. (N,H,W)."""
        probs, tgt = [], None
        for model, cfg, _thr, ck_tgt in self._loaded:
            if ck_tgt is not None and ck_tgt != self.target:
                raise ValueError(f"멤버 target={ck_tgt} != {self.target}")
            tw = cfg["data"]["time_window"]
            ms = cfg["data"].get("multiscale")
            vlist = cfg["data"].get("vars")          # 멤버별 입력채널(아노말리 등)도 ckpt cfg에서 복원
            bs = cfg["train"]["batch_size"]
            _, loader = make_loader(proc, labels, tw, self.target, years, bs, shuffle=False,
                                    multiscale=ms, vars=vlist)
            p, t = collect_probs(model, loader, self.device)
            if tgt is not None and not np.array_equal(t, tgt):
                raise ValueError("멤버 간 타깃 윈도우 불일치(time_window 차이?)")
            probs.append(p); tgt = t
        ens = np.mean(probs, axis=0) if self.combine == "mean" else probs[0]
        return ens, tgt

    def predict(self, proc, labels, years):
        """이진 탐지맵(prob>=threshold) + 확률 반환."""
        if self.threshold is None:
            raise ValueError(f"{self.target} 앙상블 임계값 미보정 — 먼저 `--calibrate` 실행 필요")
        prob, tgt = self.proba(proc, labels, years)
        return (prob >= self.threshold).astype("float32"), prob, tgt


def calibrate(target, proc, labels, manifest=MANIFEST, val_years=None):
    """앙상블 임계값을 Val에서 1회 산출해 매니페스트에 고정 기록(앙상블 전용)."""
    det = Detector.from_manifest(target, manifest)
    if det.combine != "mean":
        raise ValueError(f"{target}는 단일 모델(combine={det.combine}) — 보정 불필요"
                         "(멤버 ckpt의 val_threshold 사용).")
    if val_years is None:
        val_years = load_config("config/data.yaml")["split"]["val"]
    mask = load_land_mask(proc)
    prob, tgt = det.proba(proc, labels, val_years)
    vmask = _broadcast_mask(mask, prob.shape[0])
    thr, vcsi = best_threshold(prob, tgt, mask=vmask)
    spec = json.loads(Path(manifest).read_text())
    spec[target]["threshold"] = float(thr)
    Path(manifest).write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    print(f"[calibrate] {target} 앙상블 임계값={thr:.4f} (val_CSI={vcsi:.4f}) → {manifest} 기록")
    return thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["compound", "drought", "heatwave"])
    ap.add_argument("--proc", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--calibrate", action="store_true", help="앙상블 임계값 Val 보정 후 매니페스트 기록")
    ap.add_argument("--manifest", default=MANIFEST,
                    help="탐지기 매니페스트 경로(SPI-1 등 별도 정의는 detectors_spi1.json 지정).")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.target, args.proc, args.labels, manifest=args.manifest)
        return

    years = load_config("config/data.yaml")["split"][args.split]
    det = Detector.from_manifest(args.target, path=args.manifest)
    mask = load_land_mask(args.proc)
    binary, prob, tgt = det.predict(args.proc, args.labels, years)
    tmask = _broadcast_mask(mask, prob.shape[0])
    m = all_metrics(prob, tgt, threshold=det.threshold, mask=tmask)
    print(f"[detector] target={args.target} combine={det.combine} thr={det.threshold:.4f} "
          f"split={args.split} n={prob.shape[0]}")
    print(f"  CSI={m['CSI']:.4f} Prec={m['Precision']:.3f} Recall(POD)={m['Recall_POD']:.3f} "
          f"F1={m['F1']:.3f} TSS={m['TSS']:.3f} AUC={m['AUC_ROC']:.3f} "
          f"PR_AUC={m['PR_AUC']:.4f}(base={m['base_rate']:.4f})")


if __name__ == "__main__":
    main()
