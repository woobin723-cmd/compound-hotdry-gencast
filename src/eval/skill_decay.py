"""Pipeline B 평가 — GenCast 출력을 탐지기에 통과시켜 ERA5 진실 대비 skill 측정.

**현 단계(Stage 2 배관 검증, 2005)**: GenCast 8멤버 proc을 detector(SPI-1)에 통과시켜
멤버별·앙상블평균 탐지확률을 산출하고, ERA5 진실 라벨 대비 sanity 지표(CSI/AUC/PR-AUC)를
출력. ERA5 입력(=perfect 기준선)도 같은 detector로 통과해 GenCast와 비교한다.
※ 2005는 탐지 모델 Train 기간 → **수치는 배관 sanity일 뿐 과학적 skill 아님**.

**향후(Stage 2b, 2015-2020)**: 슬라이딩 초기조건·리드타임 Day1~15 곡선·상대 Skill Score
(ERA5=perfect·climatology=no-skill, 과거 공유분 상쇄)·Spread-Error. 아래 골격 함수 참조.

실행(cecd env):
  python -m src.eval.skill_decay --target compound --year 2005 \
      --gencast-glob 'data/gencast/gencast_2005_member*.nc' \
      --labels data/labels/labels_spi1_1989-2020.nc \
      --era5 data/processed/era5_daily_1989-2020.nc \
      --manifest config/detectors_spi1.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import VARS  # noqa: E402
from src.data.add_doy_anomaly import apply_anom_to_dataset  # noqa: E402
from src.eval.detector import Detector  # noqa: E402
from src.eval.metrics import _counts, all_metrics, reliability  # noqa: E402
from src.train import _broadcast_mask, load_land_mask  # noqa: E402

LOOKBACK_PAST = 29  # ERA5 과거일수 = time_window(30) − 1 → 첫 valid일=init(리드1), 마지막=init+14(리드15)


def _fmt(m: dict) -> str:
    return (f"CSI={m['CSI']:.4f} Prec={m['Precision']:.3f} Recall={m['Recall_POD']:.3f} "
            f"F1={m['F1']:.3f} AUC={m['AUC_ROC']:.3f} PR_AUC={m['PR_AUC']:.4f}(base={m['base_rate']:.4f})")


def evaluate_source(det: Detector, proc, labels, years, mask):
    """단일 입력장(proc)을 detector에 통과 → (prob, tgt, 지표)."""
    prob, tgt = det.proba(proc, labels, years)
    tmask = _broadcast_mask(mask, prob.shape[0]) if mask is not None else None
    m = all_metrics(prob, tgt, threshold=det.threshold, mask=tmask)
    return prob, tgt, m


def check_main(args) -> None:
    """배관 검증 모드(2005): 30일 전부 GenCast인 멤버 nc를 그대로 detector 통과."""
    if not args.gencast_glob:
        raise SystemExit("--mode check 에는 --gencast-glob 필요")
    y = args.year[0]                 # check는 단일 연도(nargs로 리스트화됨)
    years = [y, y]
    members = sorted(glob.glob(args.gencast_glob))
    if not members:
        raise FileNotFoundError(f"GenCast 멤버 없음: {args.gencast_glob}")
    print(f"[setup] target={args.target} year={y} 멤버 {len(members)}개  manifest={args.manifest}")

    det = Detector.from_manifest(args.target, path=args.manifest)
    if any("t2m_anom" in (cfg["data"].get("vars") or VARS) for _m, cfg, _thr, _t in det._loaded):
        raise SystemExit("check 모드는 아노말리(5채널) 탐지기 미지원 — leadtime 모드 사용"
                         "(t2m_anom을 하이브리드에서 즉석 산출함).")
    mask = load_land_mask(args.era5)

    # 1) GenCast 멤버별 통과 → 확률 수집
    probs, tgt = [], None
    for mp in members:
        prob, tgt_m, m = evaluate_source(det, mp, args.labels, years, mask)
        # 시간축 가드: 모든 멤버가 동일 진실라벨(=동일 평가 날짜)이어야(하루 밀림·정렬오류 배제).
        if tgt is not None and not np.array_equal(tgt, tgt_m):
            raise ValueError(f"{Path(mp).stem}: 진실라벨 불일치 — 멤버 간 평가 날짜 어긋남")
        tgt = tgt_m
        probs.append(prob)
        print(f"  [member] {Path(mp).stem:24s} {_fmt(m)}")

    # 2) 앙상블 평균(8멤버) → 지표
    ens = np.mean(probs, axis=0)
    tmask = _broadcast_mask(mask, ens.shape[0]) if mask is not None else None
    m_ens = all_metrics(ens, tgt, threshold=det.threshold, mask=tmask)
    print(f"[GenCast 앙상블({len(members)})] thr={det.threshold:.4f}  {_fmt(m_ens)}")

    # 3) 멤버 스프레드(앙상블 불확실성, sanity)
    spread = float(np.mean(np.std(probs, axis=0)))
    print(f"[GenCast spread] 멤버 표준편차 평균={spread:.4f}")

    # 4) ERA5 입력(perfect 기준선) 통과 → 비교
    _, tgt_e, m_era5 = evaluate_source(det, args.era5, args.labels, years, mask)
    if not np.array_equal(tgt, tgt_e):
        raise ValueError("ERA5 perfect 진실라벨이 GenCast와 불일치 — 평가 날짜 정렬 오류")
    print(f"[ERA5 perfect] thr={det.threshold:.4f}  {_fmt(m_era5)}")
    print(f"[배관 검증] GenCast·ERA5 동일 스키마로 detector 관통 완료 "
          f"(CSI GenCast={m_ens['CSI']:.4f} vs ERA5={m_era5['CSI']:.4f})")


# ───────────────────────── Stage 2b: 하이브리드 윈도우 리드타임 ─────────────────────────
def build_hybrid(era5_ds: xr.Dataset, future_nc: str, lookback_past: int = LOOKBACK_PAST,
                 with_anom: bool = False):
    """ERA5 과거 lookback_past일 + 미래 예보(future_nc, 15일) = 연속 proc Dataset.

    future_nc = GenCast 멤버 nc(평가 대상) 또는 ERA5 자기자신 슬라이스(perfect 기준선).
    44일(=29+15) 배열 → detector가 30일 윈도우로 슬라이딩하면 valid일 init~init+14
    (리드 1~15)가 한 번에 산출되고, 각 리드 d의 윈도우는 자동으로 ERA5(30−d)+미래(d)가 됨.
    리드15 지점에서만 정확히 ERA5 15+미래 15.

    with_anom=True(폭염·복합 아노말리 탐지기): 4채널 하이브리드(과거 ERA5+미래 GenCast)를
    먼저 만든 뒤, 연속된 t2m에 **저장된 doy 기후값·통계(Train-only)로 t2m_anom을 즉석 계산**해
    덧붙인다. GenCast nc 재생성 불필요·과거/미래 동일 공식 → 학습과 정합.
    """
    with xr.open_dataset(future_nc) as gf:        # close 보장(720회 핸들 누수 방지)
        g = gf[VARS].load()
    init = g.time.values[0]
    one = np.timedelta64(1, "D")
    past = era5_ds[VARS].sel(time=slice(init - lookback_past * one, init - one))
    if past.sizes["time"] != lookback_past:
        raise ValueError(f"ERA5 과거 {past.sizes['time']}일 != {lookback_past} "
                         f"(init={str(init)[:10]} — 연 경계/결측?)")
    hybrid = xr.concat([past, g], dim="time")
    nd = hybrid.sizes["time"]
    if nd != lookback_past + g.sizes["time"]:
        raise ValueError(f"하이브리드 길이 {nd} != {lookback_past}+{g.sizes['time']}")
    # 연속 1일 간격 검증: dataset이 배열 row를 연속일로 가정 → gap/중복이면 조용히 틀린 윈도우.
    diffs = np.diff(hybrid.time.values)
    if not np.all(diffs == one):
        bad = np.where(diffs != one)[0][:3]
        raise ValueError(f"하이브리드 time 비연속(gap/중복) at idx {bad} (init={str(init)[:10]})")
    if with_anom:
        hybrid = apply_anom_to_dataset(hybrid)    # t2m_anom 채널 추가(저장 clim·stats 재사용)
    return hybrid, init


def build_persistence(era5_ds: xr.Dataset, init, lookback_past: int, nlead: int,
                      with_anom: bool = False):
    """persistence baseline 하이브리드: 과거 lookback_past일은 ERA5 실측,
    미래 nlead일은 **init 직전 관측일(init−1)을 hold**(가장 단순한 naive 예보).
    GenCast가 이 persistence를 얼마나 능가하는지로 '예보 순기여'를 분리한다
    (하이브리드라 과거 ERA5 antecedent는 GenCast/persistence 공통이므로 상쇄)."""
    one = np.timedelta64(1, "D")
    past = era5_ds[VARS].sel(time=slice(init - lookback_past * one, init - one))
    if past.sizes["time"] != lookback_past:
        raise ValueError(f"persistence 과거 {past.sizes['time']}일 != {lookback_past}")
    last_obs = era5_ds[VARS].sel(time=init - one, drop=True)  # init 직전 관측(scalar time 제거·codex#1)
    fut_times = np.array([init + i * one for i in range(nlead)])
    future = last_obs.expand_dims(time=fut_times)            # 미래 nlead일을 last_obs로 hold
    hybrid = xr.concat([past, future.transpose("time", *last_obs.dims)], dim="time")
    if not np.all(np.diff(hybrid.time.values) == one):      # 연속성 검사(수정 후 가드)
        raise ValueError(f"persistence time 비연속(init={str(init)[:10]})")
    if hybrid.sizes["time"] != lookback_past + nlead:
        raise ValueError(f"persistence 길이 {hybrid.sizes['time']} != {lookback_past + nlead}")
    if with_anom:
        hybrid = apply_anom_to_dataset(hybrid)
    return hybrid


def build_climatology(labels_nc: str, target: str):
    """no-skill 기준선 = 각 doy·격자의 사건 기후빈도(dayofyear,H,W).

    상대 Skill Score의 no-skill 예보(=계절 기후값만 아는 예보). 라벨 전체(1989-2020)에서
    day-of-year별 평균. 희소 사건이라 확률은 낮지만 doy 계절성을 담아 0보다 강한 기준선.
    """
    with xr.open_dataset(labels_nc) as lab:
        y = lab[target].transpose("time", "lat", "lon")
        clim = y.groupby(y["time"].dt.dayofyear).mean("time")   # (dayofyear,H,W)
    return clim.load()


def _clim_stack(clim, dates):
    """valid일 리스트 → 각 날 doy의 기후빈도 맵 스택 (n,H,W)."""
    return np.stack([clim.sel(dayofyear=int(pd.Timestamp(d).dayofyear)).values for d in dates])


def _proba_hybrid(det: Detector, hybrid: xr.Dataset, labels: str):
    """하이브리드 Dataset을 임시 nc로 저장해 detector 통과 → (prob, tgt). (n_leads,H,W).

    years=None: 하이브리드 44일 전부 사용(연 경계 init에서도 [year,year] slice가 윈도우를
    잘라먹지 않게). labels는 하이브리드 time과 intersect되므로 연도 slice 불필요.
    """
    fd, tmp = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    try:
        hybrid.to_netcdf(tmp)
        return det.proba(tmp, labels, None)
    finally:
        os.unlink(tmp)


def leadtime_eval(args) -> None:
    """슬라이딩 init × 리드 Day1~15 곡선. 각 init·GenCast멤버 → 하이브리드 44일 →
    detector 통과(리드 1~15) → GenCast 8멤버 평균 → 리드별 지표 집계.
    ERA5 perfect(미래도 ERA5)·climatology no-skill 기준선도 같은 리드 격자로 산출."""
    det = Detector.from_manifest(args.target, path=args.manifest)
    era5_ds = xr.open_dataset(args.era5)
    mask = load_land_mask(args.era5)
    root = Path(args.gencast_root)
    nlead = args.n_leads
    years = args.year  # 리스트(nargs="+")

    # 하이브리드 과거 길이는 탐지기 윈도우에 의존(valid 15개 보장): lookback_past = time_window − 1.
    # 복합·가뭄 SPI-1=30→29, 폭염=14→13. 멀티스케일(multiscale)은 lookback이 별도라 미지원.
    tws = set()
    for _model, cfg, _thr, _t in det._loaded:
        if cfg["data"].get("multiscale", {}).get("enabled"):
            raise ValueError("멀티스케일 탐지기는 하이브리드 윈도우 미지원(SPI-1은 단일윈도우라 해당없음)")
        tws.add(cfg["data"]["time_window"])
    if len(tws) != 1:
        raise ValueError(f"멤버 time_window 불일치 {tws} — 하이브리드 과거길이 모호")
    lookback_past = tws.pop() - 1

    # 입력 채널: 폭염·복합 아노말리 탐지기는 t2m_anom 포함 → 하이브리드에 즉석 산출.
    all_vars = {tuple(cfg["data"].get("vars") or VARS) for _m, cfg, _thr, _t in det._loaded}
    if len(all_vars) != 1:
        raise ValueError(f"멤버 간 입력 채널 불일치 {all_vars} — 동일 채널 구성이어야 함")
    member_vars = list(all_vars.pop())
    with_anom = "t2m_anom" in member_vars
    if with_anom:
        print(f"           입력 채널={member_vars} → 하이브리드에 t2m_anom 즉석 산출(doy clim 재사용)")

    # (year, mmdd, gdir) 쌍 전부 수집 — 여러 연도를 한 리드 격자에 통합
    pairs = []
    for year in years:
        gdir = root / str(year)
        pat = re.compile(rf"gencast_{year}_(\d+)_member0\.nc$")
        inits = sorted(m.group(1) for p in gdir.glob(f"gencast_{year}_*_member0.nc")
                       if (m := pat.search(p.name)))
        if args.init:
            inits = [i for i in inits if i == args.init]
        pairs += [(year, mmdd, gdir) for mmdd in inits]
    if not pairs:
        raise FileNotFoundError(f"{root}/{{year}}/gencast_*_member0.nc 없음(years={years}, init={args.init})")
    print(f"[leadtime] target={args.target} years={years} init {len(pairs)}개")
    print(f"           lookback_past={lookback_past}(tw={lookback_past + 1}) n_leads={nlead} thr={det.threshold:.4f}")

    clim = build_climatology(args.labels, args.target)   # no-skill 기준선(doy 기후빈도)

    # 리드별 누적: GenCast 앙상블평균 / ERA5 perfect / persistence / 멤버spread / valid일
    g_prob = {d: [] for d in range(1, nlead + 1)}
    e_prob = {d: [] for d in range(1, nlead + 1)}
    p_prob = {d: [] for d in range(1, nlead + 1)}   # persistence baseline
    tgt_by_lead = {d: [] for d in range(1, nlead + 1)}
    std_by_lead = {d: [] for d in range(1, nlead + 1)}
    date_by_lead = {d: [] for d in range(1, nlead + 1)}
    spreads = []

    for year, mmdd, gdir in pairs:
        mncs = sorted(gdir.glob(f"gencast_{year}_{mmdd}_member*.nc"))
        if not mncs:
            raise FileNotFoundError(f"멤버 nc 없음: {year}/{mmdd}")
        mprobs, tgt, init0 = [], None, None
        for mnc in mncs:
            hybrid, init = build_hybrid(era5_ds, str(mnc), lookback_past, with_anom)
            # init(=valid 시작일) 명시 비교: tgt array_equal만으론 희소/all-zero 날 하루밀림 통과 가능.
            if init0 is None:
                init0 = init
            elif init != init0:
                raise ValueError(f"{mmdd} 멤버 간 init 불일치 {str(init)[:10]} != {str(init0)[:10]}")
            prob, t = _proba_hybrid(det, hybrid, args.labels)
            if prob.shape[0] != nlead:
                raise ValueError(f"{mmdd} valid {prob.shape[0]} != n_leads {nlead}")
            if tgt is not None and not np.array_equal(t, tgt):
                raise ValueError(f"{mmdd} 멤버 간 진실라벨 불일치 — 평가 날짜 어긋남")
            mprobs.append(prob); tgt = t
        ens = np.mean(mprobs, axis=0)                 # GenCast 8멤버 평균 (nlead,H,W)
        member_std = np.std(mprobs, axis=0)           # 리드별 앙상블 spread (nlead,H,W)
        spreads.append(float(np.mean(member_std)))

        # ERA5 perfect 기준선: 과거+미래 전부 ERA5 실측 = [init−lookback_past … init+14] 연속 슬라이스.
        one = np.timedelta64(1, "D")
        e_hyb = era5_ds[VARS].sel(time=slice(init - lookback_past * one,
                                             init + (nlead - 1) * one))
        if e_hyb.sizes["time"] != lookback_past + nlead:
            raise ValueError(f"{mmdd} ERA5 perfect 길이 {e_hyb.sizes['time']} "
                             f"!= {lookback_past + nlead}")
        if with_anom:
            e_hyb = apply_anom_to_dataset(e_hyb)   # perfect 기준선도 동일 채널 구성
        e_p, e_t = _proba_hybrid(det, e_hyb, args.labels)
        if not np.array_equal(e_t, tgt):
            raise ValueError(f"{mmdd} ERA5 perfect 진실라벨 불일치")

        # persistence baseline: 미래=init−1 관측 hold (naive 예보·GenCast 순기여 분리용)
        p_hyb = build_persistence(era5_ds, init, lookback_past, nlead, with_anom)
        p_p, p_t = _proba_hybrid(det, p_hyb, args.labels)
        if not np.array_equal(p_t, tgt):
            raise ValueError(f"{mmdd} persistence 진실라벨 불일치")

        for d in range(1, nlead + 1):
            g_prob[d].append(ens[d - 1])
            e_prob[d].append(e_p[d - 1])
            p_prob[d].append(p_p[d - 1])
            tgt_by_lead[d].append(tgt[d - 1])
            std_by_lead[d].append(member_std[d - 1])
            date_by_lead[d].append(init + (d - 1) * one)   # 리드 d valid일
        print(f"  [{year}/{mmdd}] init={str(init)[:10]} 멤버{len(mncs)} spread={spreads[-1]:.4f}")

    # 리드별 지표 곡선 + climatology no-skill + confusion + spread-error
    thr = det.threshold
    res = {"target": args.target, "years": years, "n_init": len(pairs),
           "lookback_past": lookback_past, "threshold": thr, "leads": list(range(1, nlead + 1)),
           "gencast": {k: [] for k in ("CSI", "AUC", "PR_AUC", "base")},
           "perfect": {k: [] for k in ("CSI", "AUC")},
           "clim": {k: [] for k in ("CSI", "AUC")},
           "persist": {k: [] for k in ("CSI", "AUC")},
           "rel_ss_csi": [], "rel_ss_auc": [], "rel_ss_persist_auc": [],
           "confusion": {k: [] for k in ("TP", "FP", "FN", "TN")},
           "spread": [], "error": []}
    print(f"\n{'lead':>4} | {'GenCast(CSI/AUC/PRAUC)':^24} | {'perf(CSI/AUC)':^15} | "
          f"{'clim(CSI/AUC)':^15} | {'persist(AUC)':^12} | {'relSS(CSI/AUC/vsPersist)':^24} | {'sprd/err':^13}")
    for d in range(1, nlead + 1):
        Pg = np.stack(g_prob[d]); Pe = np.stack(e_prob[d]); T = np.stack(tgt_by_lead[d])
        Pp = np.stack(p_prob[d])
        Pc = _clim_stack(clim, date_by_lead[d])
        tm = _broadcast_mask(mask, Pg.shape[0])
        mg = all_metrics(Pg, T, threshold=thr, mask=tm)
        me = all_metrics(Pe, T, threshold=thr, mask=tm)
        mc = all_metrics(Pc, T, threshold=thr, mask=tm)
        mp = all_metrics(Pp, T, threshold=thr, mask=tm)        # persistence
        rss_csi = relative_skill_score(mg["CSI"], me["CSI"], mc["CSI"])
        rss_auc = relative_skill_score(mg["AUC_ROC"], me["AUC_ROC"], mc["AUC_ROC"])
        # GenCast가 persistence(no-skill)를 perfect(상한)까지 얼마나 능가하나(순기여)
        rss_persist = relative_skill_score(mg["AUC_ROC"], me["AUC_ROC"], mp["AUC_ROC"])
        # confusion matrix(GenCast@thr, 90 init 합산, 육지)
        tp, fp, fn, tn = _counts(Pg >= thr, T, tm)
        # spread-error: 앙상블 spread 평균 vs RMSE(앙상블평균 확률 − 라벨), 육지만
        S = np.stack(std_by_lead[d]); m2 = mask.astype(bool) if mask is not None else None
        sprd = float(S[:, m2].mean() if m2 is not None else S.mean())
        err = float(np.sqrt(((Pg - T)[:, m2] ** 2).mean() if m2 is not None
                            else ((Pg - T) ** 2).mean()))
        for key, val in (("gencast", mg), ("perfect", me), ("clim", mc), ("persist", mp)):
            for mk, ck in zip(("CSI", "AUC", "PR_AUC", "base"),
                              ("CSI", "AUC_ROC", "PR_AUC", "base_rate")):
                if mk in res[key]:
                    res[key][mk].append(round(val[ck], 5))
        res["rel_ss_csi"].append(round(rss_csi, 4)); res["rel_ss_auc"].append(round(rss_auc, 4))
        res["rel_ss_persist_auc"].append(round(rss_persist, 4))
        for k, v in zip(("TP", "FP", "FN", "TN"), (tp, fp, fn, tn)):
            res["confusion"][k].append(v)
        res["spread"].append(round(sprd, 5)); res["error"].append(round(err, 5))
        print(f"{d:>4} | {mg['CSI']:>7.4f}{mg['AUC_ROC']:>7.3f}{mg['PR_AUC']:>9.4f} "
              f"| {me['CSI']:>7.4f}{me['AUC_ROC']:>7.3f} | {mc['CSI']:>7.4f}{mc['AUC_ROC']:>7.3f} "
              f"| {mp['AUC_ROC']:>11.3f} | {rss_csi:>7.3f}{rss_auc:>7.3f}{rss_persist:>8.3f} "
              f"| {sprd:>6.3f}{err:>7.3f}")
    # Reliability(전체 리드·init 합산): 확률 보정 곡선용 bin별 (mean_pred, obs_freq, counts)
    all_g = np.concatenate([np.stack(g_prob[d]) for d in range(1, nlead + 1)])
    all_e = np.concatenate([np.stack(e_prob[d]) for d in range(1, nlead + 1)])
    all_t = np.concatenate([np.stack(tgt_by_lead[d]) for d in range(1, nlead + 1)])
    mtot = _broadcast_mask(mask, all_g.shape[0])
    res["reliability_gencast"] = reliability(all_g, all_t, mtot)
    res["reliability_perfect"] = reliability(all_e, all_t, mtot)

    print(f"\n[spread] init 평균 멤버 표준편차 {np.mean(spreads):.4f}")
    print("[주의] 절대 CSI 직접해석 금지 — 상대 Skill Score(perfect/clim 정규화)·AUC 우선.")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[saved] {args.save_json}")


def relative_skill_score(s_model: float, s_perfect: float, s_noskill: float) -> float:
    """상대 Skill Score = (S_model − S_noskill)/(S_perfect − S_noskill).

    GenCast/Perfect(ERA5)/No-skill(climatology)이 과거 공유분을 상쇄 → 미래 15일
    GenCast 순기여만 측정. 절대 CSI 금지. no-skill=climatology는 다음 단계서 주입.
    """
    denom = s_perfect - s_noskill
    # 분모≈0(perfect와 no-skill 거의 동급)이면 불안정 → nan(codex: isclose 가드).
    return (s_model - s_noskill) / denom if not np.isclose(denom, 0.0, atol=1e-6) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="check", choices=["check", "leadtime"],
                    help="check=2005 배관검증(30일 전부 GenCast) · leadtime=하이브리드 리드곡선")
    ap.add_argument("--target", default="compound", choices=["compound", "drought", "heatwave"])
    ap.add_argument("--year", type=int, nargs="+", required=True,
                    help="leadtime: 여러 연도 통합 가능(예: 2015 2016 ... 2020). check: 단일.")
    ap.add_argument("--labels", required=True, help="ERA5 진실 라벨 nc")
    ap.add_argument("--era5", required=True, help="ERA5 proc nc(과거 채움·perfect 기준선)")
    ap.add_argument("--manifest", default="config/detectors_spi1.json")
    # check 모드
    ap.add_argument("--gencast-glob", help="check: GenCast 멤버 nc 글롭")
    # leadtime 모드
    ap.add_argument("--gencast-root", default="data/gencast",
                    help="leadtime: 이 하위 {year}/gencast_{year}_{MMDD}_member*.nc 순회.")
    ap.add_argument("--init", help="leadtime: 단일 init(MMDD)만(sanity). 미지정=전체.")
    ap.add_argument("--n-leads", type=int, default=15)
    ap.add_argument("--save-json", help="leadtime: 리드별 지표 JSON 저장 경로(시각화용).")
    args = ap.parse_args()

    if args.mode == "leadtime":
        leadtime_eval(args)
    else:
        if not args.gencast_glob:
            ap.error("--mode check 에는 --gencast-glob 필요")
        check_main(args)


if __name__ == "__main__":
    main()
