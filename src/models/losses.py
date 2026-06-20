"""손실 함수: Weighted BCE(CNN-LSTM), Focal(Transformer)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(loss, mask):
    """mask(=1 육지/0 해양)가 주어지면 육지 픽셀만 평균. 없으면 전체 평균.
    mask는 loss로 브로드캐스트 가능한 형상(예: (1,H,W))."""
    if mask is None:
        return loss.mean()
    m = mask.expand_as(loss)
    return (loss * m).sum() / m.sum().clamp(min=1.0)


def weighted_bce(logits, target, pos_weight=None, mask=None):
    """클래스 불균형 보정 BCE. pos_weight: 양성 가중치 스칼라 텐서. mask: 육지 마스크(해양 제외)."""
    loss = F.binary_cross_entropy_with_logits(
        logits, target.float(), pos_weight=pos_weight, reduction="none")
    return _masked_mean(loss, mask)


def focal_loss(logits, target, alpha=0.25, gamma=2.0, mask=None):
    """Focal Loss (Lin et al., 2017) — 희귀 클래스 대응. mask: 육지 마스크(해양 제외)."""
    target = target.float()
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).pow(gamma) * ce
    return _masked_mean(loss, mask)


def focal_tversky_loss(logits, target, alpha=0.3, beta=0.7, gamma=1.0, eps=1e-6, mask=None):
    """Focal-Tversky Loss (Abraham & Khan, 2019) — CSI(=TP/(TP+FP+FN))와 직접 정렬되는
    소프트 오버랩 손실. beta>alpha면 FN(놓침)에 더 큰 페널티 → recall 강조(희소사건 유리).
    gamma=1이면 일반 Tversky, gamma>1이면 어려운(낮은 Tversky) 사례를 상대적으로 강조.

    ⚠️ 확정 손실(Transformer=Focal, CNN-LSTM=Weighted BCE)이 기본값.
    이 손실은 config `loss.name: focal_tversky`로 명시할 때만 쓰는 **옵트인 실험용**.
    배치 전체를 하나의 집합으로 보고 soft TP/FP/FN을 집계한다. mask=육지 마스크(해양 제외).
    """
    target = target.float()
    p = torch.sigmoid(logits)
    if mask is not None:
        m = mask.expand_as(p)
        p, target = p * m, target * m
    tp = (p * target).sum()
    fp = (p * (1 - target)).sum()
    fn = ((1 - p) * target).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1 - tversky).pow(gamma)
