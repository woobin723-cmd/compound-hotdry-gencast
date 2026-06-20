"""U-Net 멤버 (앙상블 다양성 확장).

복합 극한 탐지의 본질은 **격자 픽셀별 이진 라벨** = 공간 세그멘테이션. U-Net의
encoder-decoder + skip connection은 dense 예측의 정석이며, skip이 고해상 위치 정보를
보존해 희소 양성 격자의 위치 재현에 강하다.

시간 처리: 30일×4변수를 **채널로 concat**(T*V=120채널)해 2D U-Net에 투입. 시간 순서
inductive bias는 사라지지만, ConvLSTM·CNN-LSTM(시간 순환)·Transformer(어텐션)와는
**다른 공간 중심 귀납 편향**(강한 공간 prior·국소성·skip 위치보존)을 제공 → 앙상블
다양성에 기여. ⚠️ 단, 120채널이 모든 시차를 독립 채널로 외우기 쉬워 과적합 위험은
있다(학습 후 train/val CSI gap·seed 변동성 점검 필요 — 불안정 시 base_ch↓·wd↑).

입력 (B, T, V, H, W) → 출력 (B, H, W) 로짓 (다른 멤버와 동일 인터페이스).
H,W가 2^depth로 안 나뉘므로 우/하 패딩 후 원해상도로 crop(Transformer와 동일 방식).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Conv-GN-ReLU ×2. 배치2 소배치라 BatchNorm 대신 GroupNorm(통계 배치무관)."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        g = math.gcd(groups, out_ch)
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(g, out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(g, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.body(x)


class UNet(nn.Module):
    def __init__(self, in_vars: int = 4, time_window: int = 30,
                 base_ch: int = 32, depth: int = 3, dropout: float = 0.2):
        super().__init__()
        self.depth = depth
        self.inc = DoubleConv(in_vars * time_window, base_ch)
        self.pools = nn.ModuleList()
        self.downs = nn.ModuleList()
        c = base_ch
        for _ in range(depth):
            self.pools.append(nn.MaxPool2d(2))
            self.downs.append(DoubleConv(c, c * 2))
            c *= 2
        self.upconvs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for _ in range(depth):
            self.upconvs.append(nn.ConvTranspose2d(c, c // 2, 2, stride=2))
            self.ups.append(DoubleConv(c, c // 2))   # skip(c//2) + up(c//2) = c
            c //= 2
        self.dropout = nn.Dropout2d(dropout)
        self.outc = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):  # (B, T, V, H, W)
        B, T, V, H, W = x.shape
        x = x.reshape(B, T * V, H, W)
        m = 2 ** self.depth
        Hp, Wp = math.ceil(H / m) * m, math.ceil(W / m) * m
        x = F.pad(x, (0, Wp - W, 0, Hp - H))       # 우/하 패딩(다운샘플 정합)
        skips = [self.inc(x)]
        for pool, down in zip(self.pools, self.downs):
            skips.append(down(pool(skips[-1])))
        x = skips[-1]                               # bottleneck
        for i, (upc, up) in enumerate(zip(self.upconvs, self.ups)):
            x = upc(x)
            x = up(torch.cat([skips[-2 - i], x], dim=1))
        x = self.outc(self.dropout(x))              # (B, 1, Hp, Wp)
        return x[:, 0, :H, :W]                       # 패딩 제거 → (B, H, W) 로짓
