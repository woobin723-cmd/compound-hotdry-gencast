"""탐지 모델 학습 엔트리.

Stage 0 smoke: 소데이터에서 과적합되는지(=파이프라인이 신호를 흘리는지)로 정상 판정.
  conda run -n cecd python -m src.train --config config/model_cnn_lstm.yaml \
      --proc data/processed/era5_daily_2015-07.nc \
      --labels data/labels/labels_2015-07.nc --smoke

Stage 1 본학습: 30년 통합 proc/labels를 config split(연도)으로 Train/Val/Test 분할.
  conda run -n cecd python -m src.train --config config/model_transformer.yaml \
      --proc data/processed/era5_daily_1989-2020.nc \
      --labels data/labels/labels_1989-2020.nc --full \
      --data-config config/data.yaml --out checkpoints/transformer.pt
  - val-CSI 최대 임계값을 탐색해 best-val 가중치 저장, 그 임계값으로 Test 평가.
  - land_sea_mask로 해양 격자는 평가에서 제외(데이터 자체는 NaN→0 처리됨).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, WeightedRandomSampler

# DataLoader 워커-메인 텐서 공유를 파일시스템 방식으로 전환.
# 기본(file_descriptor)은 멀티스케일(25스텝·큰 텐서) 장기 실행에서 FD를 누적해
# "Too many open files"로 죽는다(가뭄 ms 학습 30ep 완주 후 테스트 평가 단계 크래시).
torch.multiprocessing.set_sharing_strategy("file_system")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import CompoundWindowDataset  # noqa: E402
from src.eval.metrics import all_metrics, best_threshold  # noqa: E402
from src.models.cnn_lstm import CNNLSTM  # noqa: E402
from src.models.convlstm import ConvLSTM  # noqa: E402
from src.models.losses import focal_loss, focal_tversky_loss, weighted_bce  # noqa: E402
from src.models.transformer import SpatioTemporalTransformer  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.utils import load_config, resolve_path, set_seed  # noqa: E402


def seq_len(cfg: dict) -> int:
    """모델이 실제로 받는 시퀀스 길이. 멀티스케일이면 long_steps+short_days, 아니면 time_window."""
    ms = cfg["data"].get("multiscale")
    if ms and ms.get("enabled"):
        return int(ms["long_steps"]) + int(ms["short_days"])
    return int(cfg["data"]["time_window"])


def build_model(cfg: dict):
    m = cfg["model"]
    if m["name"] == "cnn_lstm":
        return CNNLSTM(
            in_vars=m["in_vars"], cnn_channels=m["cnn_channels"],
            lstm_hidden=m["lstm_hidden"], lstm_layers=m["lstm_layers"],
            bidirectional=m["bidirectional"], dropout=m["dropout"],
        )
    if m["name"] == "transformer":
        return SpatioTemporalTransformer(
            in_vars=m["in_vars"], patch_size=m["patch_size"], embed_dim=m["embed_dim"],
            depth=m["depth"], num_heads=m["num_heads"], mlp_ratio=m["mlp_ratio"],
            dropout=m["dropout"], time_window=seq_len(cfg),   # 멀티스케일이면 압축 후 길이
        )
    if m["name"] == "convlstm":
        return ConvLSTM(
            in_vars=m["in_vars"], hidden_channels=m["hidden_channels"],
            kernel_size=m.get("kernel_size", 3), dropout=m["dropout"],
        )
    if m["name"] == "unet":
        return UNet(
            in_vars=m["in_vars"], time_window=seq_len(cfg),   # T*V 채널 concat
            base_ch=m.get("base_ch", 32), depth=m.get("depth", 3), dropout=m["dropout"],
        )
    raise NotImplementedError(f"{m['name']} 미구현")


def load_land_mask(proc_nc: str, raw_dir: str = "data/raw") -> np.ndarray | None:
    """lsm.nc(land>=0.5)를 (H,W) bool 마스크로. proc 격자에 맞춰 정렬. 없으면 None."""
    mask_path = resolve_path(raw_dir) / "lsm.nc"
    if not mask_path.exists():
        print(f"[mask] {mask_path} 없음 — 해양 마스킹 없이 전체 격자 평가")
        return None
    lsm = xr.open_dataset(mask_path)["lsm"]
    # 비공간 차원(time 등) 제거 → (lat,lon)만 남김
    for d in [d for d in lsm.dims if d not in ("lat", "lon")]:
        lsm = lsm.isel({d: 0}, drop=True)
    proc = xr.open_dataset(proc_nc)
    # proc 격자에 정렬(좌표 동일 가정, 혹시 모를 정렬차 방지)
    lsm = lsm.reindex(lat=proc.lat, lon=proc.lon, method="nearest").transpose("lat", "lon")
    mask = np.asarray(lsm.values >= 0.5)
    H, W = proc.sizes["lat"], proc.sizes["lon"]
    assert mask.shape == (H, W), f"lsm 마스크 shape {mask.shape} != proc (H,W)=({H},{W})"
    return mask


def make_loader(proc, labels, tw, target, years, batch_size, shuffle, multiscale=None, vars=None):
    """config split 연도구간으로 time_slice한 DataLoader. years=[y0,y1] 또는 None."""
    ts = (str(years[0]), str(years[1])) if years else None
    ds = CompoundWindowDataset(proc, labels, time_window=tw, target=target, time_slice=ts,
                               multiscale=multiscale, vars=vars)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    return ds, loader


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """loader 전체에 대해 (prob, target) 수집. (N,H,W)."""
    model.eval()
    probs, tgts = [], []
    for xb, yb in loader:
        p = torch.sigmoid(model(xb.to(device))).cpu().numpy()
        probs.append(p)
        tgts.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(tgts)


def _broadcast_mask(mask, n):
    """(H,W) → (N,H,W). mask가 None이면 None."""
    return None if mask is None else np.broadcast_to(mask, (n, *mask.shape))


def compute_loss(logits, yb, cfg, pos_w, mask=None):
    lc = cfg.get("loss", {})
    name = lc.get("name")
    if name == "focal":
        return focal_loss(logits, yb, lc["alpha"], lc["gamma"], mask=mask)
    if name == "focal_tversky":  # 옵트인 실험용(확정 손실 아님)
        return focal_tversky_loss(logits, yb, lc.get("alpha", 0.3),
                                  lc.get("beta", 0.7), lc.get("gamma", 1.0), mask=mask)
    return weighted_bce(logits, yb, pos_weight=pos_w, mask=mask)


def build_scheduler(opt, cfg, epochs):
    """warmup(선형) → cosine 감쇠. sched.step()은 epoch 종료 후 호출되므로
    학습 epoch e(1-기반)는 lr_lambda(e-1)을 쓴다 → e=epochs에서 cosine이 0에 도달(off-by-one 보정)."""
    warmup = int(cfg["train"].get("warmup_epochs", 0))

    def lr_lambda(idx):  # idx: 0-기반(=학습 epoch e-1)
        e = idx + 1
        if warmup > 0 and e <= warmup:
            return e / warmup
        prog = (e - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def build_train_sampler(tr_ds, cfg, land_mask):
    """오버샘플 WeightedRandomSampler. config train.sampler.oversample=true일 때만.
    육지 양성 픽셀 수에 비례해 가중(강한 사건일수록 자주 노출) — 모든 윈도우 baseline 1 유지.
    이진 비율(pos_target) 방식은 실데이터 양성-윈도우가 이미 다수(~66%)라 역효과여서 폐기.
    pos_boost: 최다 양성 윈도우의 가중을 (1+pos_boost)배까지."""
    scfg = cfg["train"].get("sampler", {}) or {}
    if not scfg.get("oversample"):
        return None
    cnt = tr_ds.pos_per_window(land_mask).astype(np.float64)  # 육지 양성 픽셀 수
    cmax = float(cnt.max())
    if cmax <= 0:
        print("[sampler] 오버샘플 건너뜀(육지 양성 윈도우 없음)")
        return None
    boost = float(scfg.get("pos_boost", 4.0))
    w = 1.0 + boost * (cnt / cmax)             # baseline 1 + 양성밀도 비례 가중
    n_posw = int((cnt > 0).sum())
    print(f"[sampler] 양성밀도 오버샘플 pos_boost={boost} "
          f"육지양성윈도우={n_posw}/{len(cnt)} 최대가중={1.0 + boost:.1f}x")
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)


def train_full(cfg, args, device):
    """Stage 1 본학습: Train/Val/Test 분할 + val-CSI 최대 임계값 + best-val 저장."""
    dcfg = load_config(args.data_config)["split"]
    tw = cfg["data"]["time_window"]
    ms = cfg["data"].get("multiscale")
    vlist = cfg["data"].get("vars")          # doy-아노말리 등 입력채널 확장(기본 None→VARS 4)
    bs = cfg["train"]["batch_size"]
    epochs = cfg["train"]["epochs"]

    tr_ds, tr_loader = make_loader(args.proc, args.labels, tw, args.target, dcfg["train"], bs, True, ms, vlist)
    va_ds, va_loader = make_loader(args.proc, args.labels, tw, args.target, dcfg["val"], bs, False, ms, vlist)
    te_ds, te_loader = make_loader(args.proc, args.labels, tw, args.target, dcfg["test"], bs, False, ms, vlist)
    # 육지 마스크 먼저 로드 → 손실/통계/샘플러를 평가(육지만)와 일치시킴
    mask = load_land_mask(args.proc)
    mask_t = None if mask is None else torch.as_tensor(
        mask.astype(np.float32), device=device).unsqueeze(0)  # (1,H,W) 브로드캐스트용

    pw_raw = tr_ds.pos_weight_masked(mask)  # 육지만으로 계산(평가와 일치)
    win_desc = (f"multiscale lookback={tr_ds.lookback}d→{tr_ds.n_steps}steps"
                if tr_ds.ms else f"window={tw}")
    print(f"[data] target={args.target} train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} "
          f"{win_desc} pos_weight(육지)={pw_raw:.1f}")

    # 양성밀도 오버샘플링(옵트인, 육지 기준): sampler를 쓰면 shuffle 비활성
    sampler = build_train_sampler(tr_ds, cfg, mask)
    if sampler is not None:
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=sampler, num_workers=2)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    sched = build_scheduler(opt, cfg, epochs)
    grad_clip = cfg["train"].get("grad_clip")  # None이면 클리핑 안 함

    # pos_weight 캡: 오버샘플과 함께 쓸 때 과도한 양성가중(→오탐 폭증) 완화. Weighted BCE 한정.
    cap = cfg.get("loss", {}).get("pos_weight_cap")
    pw_val = pw_raw if cap is None else min(pw_raw, float(cap))
    pos_w = torch.tensor(pw_val, device=device)
    if cfg.get("loss", {}).get("name") in (None, "weighted_bce"):
        print(f"[loss] weighted_bce pos_weight={pw_val:.1f} (raw={pw_raw:.1f}, cap={cap})")

    # 조기종료: val-CSI가 patience epoch 동안 개선 없으면 중단
    es = cfg["train"].get("early_stopping", {}) or {}
    patience = es.get("patience")
    no_improve = 0

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val_csi, best_thr, best_ep = -1.0, 0.5, 0

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = compute_loss(model(xb), yb, cfg, pos_w, mask=mask_t)
            opt.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            tot += loss.item() * xb.size(0)
        avg = tot / len(tr_ds)
        if sched is not None:
            sched.step()

        # --- val 평가: val-CSI 최대 임계값 탐색 → best 갱신 시 체크포인트 저장 ---
        vp, vt = collect_probs(model, va_loader, device)
        vmask = _broadcast_mask(mask, vp.shape[0])
        thr, vcsi = best_threshold(vp, vt, mask=vmask)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            vm = all_metrics(vp, vt, threshold=thr, mask=vmask)
            print(f"[ep {ep:3d}] train_loss={avg:.4f} val_CSI={vcsi:.4f}@thr={thr:.2f} "
                  f"val_F1={vm['F1']:.3f} val_AUC={vm['AUC_ROC']:.3f}")
        # best 갱신 시(또는 아직 한 번도 저장 안 됐으면 fallback으로) 체크포인트 저장
        improved = (not np.isnan(vcsi)) and vcsi > best_val_csi
        if improved or not out.exists():
            if improved:
                best_val_csi, best_thr, best_ep = vcsi, thr, ep
                no_improve = 0
            torch.save({"model": model.state_dict(), "config": cfg, "seed": cfg["train"]["seed"],
                        "target": args.target, "val_threshold": best_thr,
                        "val_csi": best_val_csi}, out)
        else:
            no_improve += 1  # NaN(val 양성 없음) 포함 — 조기종료가 작동하도록

        # --- 조기종료 판정 ---
        if patience and no_improve >= patience:
            print(f"[early-stop] {patience}ep 동안 val-CSI 개선 없음 → ep{ep}에서 중단 "
                  f"(best ep{best_ep} CSI={best_val_csi:.4f})")
            break

    # --- best-val 가중치로 Test 평가(val에서 고른 임계값 적용) ---
    ckpt = torch.load(out, map_location=device)
    model.load_state_dict(ckpt["model"])
    tp, tt = collect_probs(model, te_loader, device)
    tmask = _broadcast_mask(mask, tp.shape[0])
    test_m = all_metrics(tp, tt, threshold=ckpt["val_threshold"], mask=tmask)
    print(f"[best-val] CSI={best_val_csi:.4f} @thr={best_thr:.2f} (ep{best_ep})")
    print(f"[TEST] thr={ckpt['val_threshold']:.2f} {test_m}")
    print(f"[saved] {out}")


def train_smoke(cfg, args, device):
    """Stage 0 smoke: 단일 파일, 학습셋 과적합으로 파이프라인 신호 확인."""
    tw = cfg["data"]["time_window"]
    ms = cfg["data"].get("multiscale")          # smoke도 본학습과 동일 윈도우 구성(검증 유효성)
    vlist = cfg["data"].get("vars")
    bs = cfg["smoke"]["batch_size"]
    epochs = cfg["smoke"]["epochs"]

    ds = CompoundWindowDataset(args.proc, args.labels, time_window=tw, target=args.target,
                               multiscale=ms, vars=vlist)
    win_desc = (f"multiscale lookback={ds.lookback}d→{ds.n_steps}steps" if ds.ms
                else f"window={tw}")
    print(f"[data] target={args.target} samples={len(ds)} {win_desc} pos_weight={ds.pos_weight:.1f}")
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    pos_w = torch.tensor(ds.pos_weight, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = compute_loss(model(xb), yb, cfg, pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        avg = tot / len(ds)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[ep {ep:3d}] train_loss={avg:.4f}")

    prob, tgt = collect_probs(model, DataLoader(ds, batch_size=bs), device)
    print(f"[train-set metrics] {all_metrics(prob, tgt)}")
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "seed": cfg["train"]["seed"]}, out)
    print(f"[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/model_cnn_lstm.yaml")
    ap.add_argument("--proc", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true", help="Stage 1 본학습(Train/Val/Test 분할)")
    ap.add_argument("--data-config", default="config/data.yaml", help="split 연도 출처")
    ap.add_argument("--target", default="compound",
                    help="compound(기본) | heatwave | drought. Stage0 슬라이스는 heatwave 권장")
    ap.add_argument("--out", default="checkpoints/cnn_lstm.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    if args.full:
        train_full(cfg, args, device)
    else:
        train_smoke(cfg, args, device)


if __name__ == "__main__":
    main()
