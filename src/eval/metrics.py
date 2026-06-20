"""탐지 지표: CSI(주), Precision, Recall(POD), F1, TSS(PSS), ETS, AUC-ROC, PR-AUC, Brier.
(Reliability는 별도 plot)

희소 이벤트(복합 0.19%)에서 CSI 절대값이 낮아 보이는 한계를 보완하기 위해 지표를 분해·보강한다:
- Precision/Recall(POD): CSI를 오탐/미탐 축으로 분해.
- TSS(=PSS, Peirce Skill Score): POD−POFD. 불균형에 강건(무작위=0, 완벽=1).
- PR-AUC(Average Precision): 희귀 이벤트 식별력. 무작위 기준선=base rate라 ROC-AUC보다 정직.
"""
from __future__ import annotations

import numpy as np


def _counts(pred_bin, target, mask=None):
    p = pred_bin.astype(bool).ravel()
    t = target.astype(bool).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    tn = int(np.sum(~p & ~t))
    return tp, fp, fn, tn


def csi(pred_bin, target, mask=None) -> float:
    """Critical Success Index = TP/(TP+FP+FN). 주 평가지표."""
    tp, fp, fn, _ = _counts(pred_bin, target, mask)
    denom = tp + fp + fn
    return tp / denom if denom else float("nan")


def f1(pred_bin, target, mask=None) -> float:
    tp, fp, fn, _ = _counts(pred_bin, target, mask)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else float("nan")


def precision(pred_bin, target, mask=None) -> float:
    """Precision = TP/(TP+FP). 양성예측 중 적중 비율(오탐 확인). FAR=1−Precision."""
    tp, fp, _, _ = _counts(pred_bin, target, mask)
    denom = tp + fp
    return tp / denom if denom else float("nan")


def recall(pred_bin, target, mask=None) -> float:
    """Recall = POD(Probability of Detection) = TP/(TP+FN). 실제 사건 중 포착 비율(미탐 확인)."""
    tp, _, fn, _ = _counts(pred_bin, target, mask)
    denom = tp + fn
    return tp / denom if denom else float("nan")


def tss(pred_bin, target, mask=None) -> float:
    """True Skill Statistic = Peirce Skill Score = POD − POFD = TP/(TP+FN) − FP/(FP+TN).
    불균형에 강건(우연=0, 완벽=1, 역예측=−1). 기상/기후 표준 skill score."""
    tp, fp, fn, tn = _counts(pred_bin, target, mask)
    pod_d = tp + fn          # 실제 양성
    pofd_d = fp + tn         # 실제 음성
    if pod_d == 0 or pofd_d == 0:
        return float("nan")
    return tp / pod_d - fp / pofd_d


def pr_auc(prob, target, mask=None) -> float:
    """PR-AUC를 Average Precision(AP)으로 추정. 희귀 이벤트 식별력(무작위 기준선=base rate).

    AP = Σ(Rₙ−Rₙ₋₁)Pₙ. sklearn 권장 추정(사다리꼴 `auc(recall,prec)`는 선형보간 과대평가
    위험이 있어 미사용). 엄밀히 'PR-curve 사다리꼴 적분'과는 미세 차이가 있으나 실무 표준.
    ROC-AUC는 불균형에서 부풀 수 있으나 AP는 base rate 대비라 더 정직.
    단일클래스(전부 양성/전부 음성)면 정의 불가 → nan(auc_roc와 일관)."""
    from sklearn.metrics import average_precision_score

    p, t = prob.ravel(), target.astype(int).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    if t.size == 0 or t.min() == t.max():    # 양성·음성 한쪽만 → 정의 불가
        return float("nan")
    return float(average_precision_score(t, p))


def base_rate(target, mask=None) -> float:
    """양성 라벨 비율(=무작위 PR-AUC 기준선). PR-AUC/base_rate가 클수록 강한 skill."""
    t = target.astype(float).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        t = t[m]
    return float(t.mean()) if t.size else float("nan")


def ets(pred_bin, target, mask=None) -> float:
    """Equitable Threat Score (우연성 보정)."""
    tp, fp, fn, tn = _counts(pred_bin, target, mask)
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    hits_rand = (tp + fp) * (tp + fn) / n
    denom = tp + fp + fn - hits_rand
    return (tp - hits_rand) / denom if denom else float("nan")


def brier(prob, target, mask=None) -> float:
    p, t = prob.ravel(), target.astype(float).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    return float(np.mean((p - t) ** 2))


def auc_roc(prob, target, mask=None) -> float:
    from sklearn.metrics import roc_auc_score

    p, t = prob.ravel(), target.astype(int).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    if t.size == 0 or t.min() == t.max():   # 빈 마스크·단일클래스 → 정의 불가
        return float("nan")
    return float(roc_auc_score(t, p))


def all_metrics(prob, target, threshold=0.5, mask=None) -> dict:
    pred_bin = prob >= threshold
    return {
        "CSI": csi(pred_bin, target, mask),
        "Precision": precision(pred_bin, target, mask),
        "Recall_POD": recall(pred_bin, target, mask),
        "F1": f1(pred_bin, target, mask),
        "TSS": tss(pred_bin, target, mask),
        "ETS": ets(pred_bin, target, mask),
        "AUC_ROC": auc_roc(prob, target, mask),
        "PR_AUC": pr_auc(prob, target, mask),
        "Brier": brier(prob, target, mask),
        "base_rate": base_rate(target, mask),
    }


def best_threshold(prob, target, mask=None) -> tuple[float, float]:
    """val-CSI를 최대화하는 임계값 탐색 → (best_thr, best_csi).

    정렬 후 누적합 스윕으로 **정확한** 최적 임계값을 O(n log n)에 찾는다
    (고정 그리드는 최적값을 놓칠 수 있음). 본학습은 0.5 고정 대신 이 임계값을
    Test에 적용한다(불균형 라벨에서 0.5는 보통 비최적).

    prob>=thr 예측 기준. 양성 라벨이 없거나 유효표본이 없으면 (0.5, nan).
    non-finite(prob/target)는 제외(라벨 int8·prob sigmoid라 실파이프라인엔 없지만 방어).
    """
    p = prob.ravel().astype(np.float64)
    t = target.ravel().astype(np.float64)
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    finite = np.isfinite(p) & np.isfinite(t)
    p, t = p[finite], (t[finite] > 0.5)
    total_pos = int(t.sum())
    if p.size == 0 or total_pos == 0:
        return 0.5, float("nan")
    order = np.argsort(-p)                 # 확률 내림차순
    ps, ts = p[order], t[order].astype(np.int64)
    tp_cum = np.cumsum(ts)                 # 상위 k개를 양성예측 시 TP
    k = np.arange(1, ps.size + 1)          # 양성예측 개수 = TP+FP
    denom = k + total_pos - tp_cum         # TP+FP+FN
    csi_arr = tp_cum / denom
    best_i = int(np.argmax(csi_arr))
    return float(ps[best_i]), float(csi_arr[best_i])


def reliability(prob, target, mask=None, n_bins: int = 10) -> dict:
    """신뢰도 다이어그램용 bin별 (평균 예측확률, 관측빈도, 표본수).

    완벽 보정이면 mean_pred ≈ obs_freq. plot은 별도(notebook)에서.
    """
    p = prob.ravel()
    t = target.astype(float).ravel()
    if mask is not None:
        m = mask.astype(bool).ravel()
        p, t = p[m], t[m]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        counts.append(n)
        mean_pred.append(float(p[sel].mean()) if n else float("nan"))
        obs_freq.append(float(t[sel].mean()) if n else float("nan"))
    return {"mean_pred": mean_pred, "obs_freq": obs_freq, "counts": counts}
