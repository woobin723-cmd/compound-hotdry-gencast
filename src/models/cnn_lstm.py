"""CNN-LSTM 베이스라인.

CNN(공간 패턴) → 프레임별 임베딩 → BiLSTM(시간 연속성) → 격자별 분류.
입력 (B, T, V, H, W) → 출력 (B, H, W) 복합 Hot-Dry 로짓.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_vars: int, channels: list[int]):
        super().__init__()
        layers = []
        c_prev = in_vars
        for c in channels:
            layers += [nn.Conv2d(c_prev, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_prev = c
        self.body = nn.Sequential(*layers)
        self.out_ch = c_prev

    def forward(self, x):  # (B*T, V, H, W)
        return self.body(x)


class CNNLSTM(nn.Module):
    def __init__(self, in_vars=4, cnn_channels=(32, 64, 128),
                 lstm_hidden=256, lstm_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.encoder = CNNEncoder(in_vars, list(cnn_channels))
        feat = self.encoder.out_ch
        self.lstm = nn.LSTM(feat, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Conv2d(lstm_out, 64, 1), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):  # (B, T, V, H, W)
        B, T, V, H, W = x.shape
        feat = self.encoder(x.reshape(B * T, V, H, W))          # (B*T, C, H, W)
        C = feat.shape[1]
        feat = feat.reshape(B, T, C, H, W).permute(0, 3, 4, 1, 2)  # (B,H,W,T,C)
        feat = feat.reshape(B * H * W, T, C)
        out, _ = self.lstm(feat)                                 # (B*H*W, T, lstm_out)
        last = out[:, -1, :].reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return self.head(last).squeeze(1)                        # (B, H, W) 로짓
