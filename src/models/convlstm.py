"""ConvLSTM 멤버 (앙상블 다양성 확장).

CNN-LSTM 베이스라인은 CNN으로 공간을 임베딩한 뒤 격자를 (B*H*W, T, C)로 펴서 **각
격자점을 독립적으로** LSTM에 넣는다 → 시간 순환 단계에서 이웃 격자 간 상호작용이 없다.
ConvLSTM은 LSTM의 게이트 연산을 **Conv2d**로 바꿔, hidden/cell state를 (C,H,W) 텐서로
유지한 채 시간을 순환한다 → 매 시간스텝마다 공간 이웃과 상호작용(공간 상관 보존).

입력 (B, T, V, H, W) → 출력 (B, H, W) 로짓 (CNN-LSTM·Transformer와 동일 인터페이스).
설계: 작은 데이터(동아시아 30년)·극희소 사건 → 과적합 억제 위해 얕게(2층·hidden 64).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """입력·은닉을 합쳐 4게이트(i,f,o,g)를 Conv2d 한 번으로 산출."""

    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3):
        super().__init__()
        assert kernel % 2 == 1, f"kernel은 홀수여야 공간 크기 보존(kernel={kernel})"
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=kernel // 2)
        # forget gate(두 번째 청크) bias=1 초기화 — 장기 의존 안정화(LSTM 표준 관행)
        nn.init.zeros_(self.conv.bias)
        self.conv.bias.data[hid_ch:2 * hid_ch].fill_(1.0)

    def forward(self, x, h, c):  # x,h,c: (B, *, H, W)
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, in_vars: int = 4, hidden_channels=(64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        self.cells = nn.ModuleList()
        c_prev = in_vars
        for hc in hidden_channels:
            self.cells.append(ConvLSTMCell(c_prev, hc, kernel_size))
            c_prev = hc
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Sequential(
            nn.Conv2d(c_prev, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):  # (B, T, V, H, W)
        B, T, V, H, W = x.shape
        h = [torch.zeros(B, cell.hid_ch, H, W, device=x.device, dtype=x.dtype) for cell in self.cells]
        c = [torch.zeros_like(hi) for hi in h]
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(inp, h[i], c[i])
                inp = h[i]
        out = self.dropout(h[-1])                 # 마지막 시간스텝 최상위 은닉 (B, C, H, W)
        return self.head(out).squeeze(1)          # (B, H, W) 로짓
