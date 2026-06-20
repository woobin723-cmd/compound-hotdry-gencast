"""앙상블 평가: 학습된 CNN-LSTM + Transformer의 확률을 평균해 Test CSI를 끌어올린다.

후처리 단계의 무비용 스킬 향상(학습 불필요). 각 모델 확률을 Val에서 평균→val-CSI 최대
임계값 탐색→Test에 적용. 개별 모델 Test 지표도 함께 출력(비교용).

  conda run -n cecd python -m scripts.ensemble_eval \
      --proc data/processed/era5_daily_1989-2020.nc \
      --labels data/labels/labels_1989-2020.nc \
      --ckpts checkpoints/cnn_lstm_formal.pt checkpoints/transformer_formal.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.metrics import all_metrics, best_threshold  # noqa: E402
from src.train import (build_model, collect_probs, load_land_mask,  # noqa: E402
                       make_loader, _broadcast_mask)
from src.utils import load_config  # noqa: E402


def model_probs(ckpt_path, proc, labels, target, years, device):
    """체크포인트 로드 → 해당 split의 (prob, target) 수집. (cfg time_window 사용)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    tw = cfg["data"]["time_window"]
    ms = cfg["data"].get("multiscale")          # 학습 때 멀티스케일이면 평가도 동일 구성
    vlist = cfg["data"].get("vars")             # 5채널(doy-아노말리) ckpt도 동일 채널로 로드
    bs = cfg["train"]["batch_size"]
    _, loader = make_loader(proc, labels, tw, target, years, bs, shuffle=False, multiscale=ms, vars=vlist)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    p, t = collect_probs(model, loader, device)
    return p, t, float(ckpt.get("val_threshold", 0.5)), ckpt.get("target")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True, help="앙상블할 체크포인트들")
    ap.add_argument("--data-config", default="config/data.yaml")
    ap.add_argument("--target", default="compound")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = load_config(args.data_config)["split"]
    mask = load_land_mask(args.proc)

    val_probs, test_probs = [], []
    val_tgt = test_tgt = None
    for ck in args.ckpts:
        vp, vt, vthr, ck_tgt = model_probs(ck, args.proc, args.labels, args.target, split["val"], device)
        tp, tt, _, _ = model_probs(ck, args.proc, args.labels, args.target, split["test"], device)
        # 학습 타깃 일치 검증(다른 라벨용 모델 섞임 방지)
        if ck_tgt is not None and ck_tgt != args.target:
            raise ValueError(f"{Path(ck).name} 학습 target={ck_tgt} != 평가 target={args.target}")
        # 타깃 라벨 배열이 모델 간 동일한지 검증(윈도우 정합)
        if val_tgt is not None:
            assert np.array_equal(vt, val_tgt) and np.array_equal(tt, test_tgt), \
                f"{Path(ck).name} 타깃 라벨이 다른 모델과 불일치(time_window 차이?)"
        # 개별 모델 Test 지표(각자 val 임계값)
        tmask = _broadcast_mask(mask, tp.shape[0])
        m = all_metrics(tp, tt, threshold=vthr, mask=tmask)
        print(f"[개별] {Path(ck).name} target={ck_tgt} thr={vthr:.2f} "
              f"CSI={m['CSI']:.4f} Prec={m['Precision']:.3f} Recall(POD)={m['Recall_POD']:.3f} "
              f"F1={m['F1']:.3f} TSS={m['TSS']:.3f} AUC_ROC={m['AUC_ROC']:.3f} "
              f"PR_AUC={m['PR_AUC']:.4f}(base={m['base_rate']:.4f})")
        val_probs.append(vp)
        test_probs.append(tp)
        val_tgt, test_tgt = vt, tt

    # 형상 정합 확인 후 확률 평균
    shapes_v = {p.shape for p in val_probs}
    shapes_t = {p.shape for p in test_probs}
    assert len(shapes_v) == 1 and len(shapes_t) == 1, f"확률 형상 불일치 v={shapes_v} t={shapes_t}"
    ens_val = np.mean(val_probs, axis=0)
    ens_test = np.mean(test_probs, axis=0)

    # Val에서 앙상블 best 임계값 → Test 적용
    vmask = _broadcast_mask(mask, ens_val.shape[0])
    tmask = _broadcast_mask(mask, ens_test.shape[0])
    ethr, evcsi = best_threshold(ens_val, val_tgt, mask=vmask)
    em = all_metrics(ens_test, test_tgt, threshold=ethr, mask=tmask)
    print(f"[앙상블] n={len(args.ckpts)} val_best_thr={ethr:.2f} val_CSI={evcsi:.4f}")
    print(f"[앙상블 TEST] thr={ethr:.2f} {em}")


if __name__ == "__main__":
    main()
