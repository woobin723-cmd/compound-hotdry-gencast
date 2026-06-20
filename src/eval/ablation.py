"""앙상블 멤버 ablation (2026-06-08 멤버 확장 후 최적 조합 선정).

4멤버(CNN-LSTM·ConvLSTM·U-Net·Transformer, 전부 Focal·tw30) 단독/조합을 비교해
**최적 앙상블 멤버셋**을 데이터로 고른다. 앙상블 이득은 멤버 수가 아니라 오차
탈상관에서 나오므로, 약한·중복 멤버는 단순평균+단일threshold를 끌어내릴 수 있다.

절차(누수 차단): 멤버별 확률을 Val·Test에서 1회씩만 계산해 캐시 → 각 조합은
  ① Val에서 멤버확률 평균 → best_threshold(육지마스크) 보정
  ② 그 threshold로 Test 평가(all_metrics)
즉 threshold는 항상 Val에서만 정해지고 Test는 평가에만 쓴다.

부가: 멤버쌍 오차상관(Test 확률, 육지)·결과 JSON 저장.
실행: conda run -n cecd python -m src.eval.ablation \
        --proc data/processed/era5_daily_1989-2020.nc \
        --labels data/labels/labels_spi1_1989-2020.nc
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from src.eval.detector import _load_member
from src.eval.metrics import all_metrics, best_threshold
from src.train import _broadcast_mask, collect_probs, load_land_mask, make_loader
from src.utils import load_config

# 멤버 약칭 → (표시명, ckpt 경로 빌더). 전부 Focal·tw30. 타깃(compound/drought/heatwave)별로
# 동일 4멤버 구조를 공유하므로 경로를 target에서 도출. transformer만 *_spi1(focal 내장),
# 나머지는 *_spi1_focal. (2026-06-13: 폭염·가뭄 앙상블 확장에 맞춰 compound 하드코딩 → target 도출.)
def members_for(target):
    return {
        "C": ("CNN-LSTM", f"checkpoints/cnn_lstm_{target}_spi1_focal.pt"),
        "V": ("ConvLSTM", f"checkpoints/convlstm_{target}_spi1_focal.pt"),
        "U": ("U-Net", f"checkpoints/unet_{target}_spi1_focal.pt"),
        "T": ("Transformer", f"checkpoints/transformer_{target}_spi1.pt"),
    }


def _member_probs(members, key, proc, labels, target, years, device):
    """단일 멤버의 (prob, tgt) — make_loader는 멤버 ckpt config의 윈도우로 구성."""
    model, cfg, _thr, ck_tgt = _load_member(members[key][1], device)
    if ck_tgt != target:
        raise ValueError(f"{key} target={ck_tgt} != {target}")
    tw = cfg["data"]["time_window"]
    ms = cfg["data"].get("multiscale")
    vlist = cfg["data"].get("vars")
    bs = cfg["train"]["batch_size"]
    _, loader = make_loader(proc, labels, tw, target, years, bs, shuffle=False, multiscale=ms, vars=vlist)
    return collect_probs(model, loader, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--target", default="compound",
                    help="라벨/평가 타깃(heatwave/drought/compound). 체크포인트 내부 target 검증에도 사용.")
    ap.add_argument("--members-target", default=None,
                    help="멤버 ckpt 경로 도출용 문자열(기본=--target). 예: 'heatwave_anom'이면 "
                         "checkpoints/{model}_heatwave_anom_spi1_focal.pt 사용(라벨 타깃은 heatwave 유지).")
    ap.add_argument("--out", default=None,
                    help="기본값 None → data/skill_json/ablation_{members_target}_spi1.json (덮어쓰기 방지)")
    ap.add_argument("--min-size", type=int, default=1)
    args = ap.parse_args()
    mtgt = args.members_target or args.target
    if args.out is None:
        args.out = f"data/skill_json/ablation_{mtgt}_spi1.json"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = load_config("config/data.yaml")["split"]
    val_years, test_years = split["val"], split["test"]
    mask = load_land_mask(args.proc)

    members = members_for(mtgt)        # 경로는 members-target, 라벨/ck_tgt 검증은 args.target
    keys = list(members)
    pv, pt = {}, {}        # 멤버별 Val/Test 확률 캐시
    tgt_v = tgt_t = None
    for k in keys:
        print(f"[probs] {members[k][0]} 확률 계산(Val·Test)…", flush=True)
        pv[k], tv = _member_probs(members, k, args.proc, args.labels, args.target, val_years, device)
        pt[k], tt = _member_probs(members, k, args.proc, args.labels, args.target, test_years, device)
        if tgt_v is None:
            tgt_v, tgt_t = tv, tt
        elif not (np.array_equal(tv, tgt_v) and np.array_equal(tt, tgt_t)):
            raise ValueError(f"{k} 타깃 불일치(time_window/split 차이?)")  # -O서도 유지

    vmask = _broadcast_mask(mask, tgt_v.shape[0])
    tmask = _broadcast_mask(mask, tgt_t.shape[0])
    if tmask is None:
        raise ValueError("land mask 없음(lsm.nc 필요) — 해양 마스킹 없는 평가는 무의미")

    # 멤버쌍 예측 확률상관(Test 확률, 육지 격자만) — 낮을수록 예측 다양성↑(앙상블 이득).
    # 주의: 라벨 대비 residual error 상관이 아니라 확률 자체의 상관(다양성 지표).
    land = tmask.astype(bool)
    corr = {}
    for a, b in combinations(keys, 2):
        c = float(np.corrcoef(pt[a][land], pt[b][land])[0, 1])
        corr[f"{a}-{b}"] = round(c, 4)

    # 모든 조합(크기 min_size..4) Val 보정 → Test 평가
    rows = []
    for r in range(args.min_size, len(keys) + 1):
        for combo in combinations(keys, r):
            ev = np.mean([pv[k] for k in combo], axis=0)
            et = np.mean([pt[k] for k in combo], axis=0)
            thr, vcsi = best_threshold(ev, tgt_v, mask=vmask)
            m = all_metrics(et, tgt_t, threshold=thr, mask=tmask)
            rows.append({
                "members": "".join(combo), "size": r, "thr": round(float(thr), 4),
                "val_CSI": round(float(vcsi), 4), "test_CSI": round(m["CSI"], 4),
                "Prec": round(m["Precision"], 4), "Recall": round(m["Recall_POD"], 4),
                "TSS": round(m["TSS"], 4), "AUC": round(m["AUC_ROC"], 4),
                "PR_AUC": round(m["PR_AUC"], 4), "Brier": round(m["Brier"], 5),
            })

    rows.sort(key=lambda x: x["test_CSI"], reverse=True)
    base = all_metrics(tgt_t.astype("float32"), tgt_t, threshold=0.5, mask=tmask)["base_rate"]

    print(f"\n=== Ablation (target={args.target}, base_rate={base:.4f}, N_test={tgt_t.shape[0]}) ===")
    print(f"{'members':9s} {'sz':>2s} {'thr':>6s} {'valCSI':>7s} {'tstCSI':>7s} "
          f"{'Prec':>6s} {'Rec':>6s} {'TSS':>6s} {'AUC':>6s} {'PRAUC':>6s}")
    for x in rows:
        print(f"{x['members']:9s} {x['size']:>2d} {x['thr']:>6.3f} {x['val_CSI']:>7.4f} "
              f"{x['test_CSI']:>7.4f} {x['Prec']:>6.3f} {x['Recall']:>6.3f} {x['TSS']:>6.3f} "
              f"{x['AUC']:>6.3f} {x['PR_AUC']:>6.4f}")

    print("\n=== 멤버쌍 예측 확률상관(낮을수록 다양성↑·앙상블 이득↑) ===")
    for k, v in sorted(corr.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.4f}")

    # 최종 선택은 Val 기준이 엄밀(Test로 고르면 selection bias). 둘 다 보고해 일치 여부 확인.
    best_test = rows[0]
    best_val = max(rows, key=lambda x: x["val_CSI"])
    print(f"\n[BEST by val_CSI ] {best_val['members']} (size {best_val['size']}) "
          f"val_CSI={best_val['val_CSI']:.4f} → test_CSI={best_val['test_CSI']:.4f} thr={best_val['thr']:.4f}")
    print(f"[BEST by test_CSI] {best_test['members']} (size {best_test['size']}) "
          f"test_CSI={best_test['test_CSI']:.4f} (참고용·selection bias 주의)")
    if best_val["members"] == best_test["members"]:
        print("  → Val·Test 최적 조합 일치(선택 견고).")

    out = {"target": args.target, "base_rate": round(float(base), 5),
           "legend": {k: members[k][0] for k in keys},
           "rows": rows, "pair_corr_note": "Test 육지격자 확률상관(residual error 아님)",
           "pair_corr": corr, "best_by_val": best_val, "best_by_test": best_test}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
