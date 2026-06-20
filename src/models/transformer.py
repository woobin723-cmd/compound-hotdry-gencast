"""Spatio-Temporal Transformer (주력 모델, Focal Loss).

패치 토큰화(8×8) → factorized(공간/시간 분리) 어텐션 → 패치→픽셀 디코딩.
입력 (B, T, V, H, W) → 출력 (B, H, W) 복합 Hot-Dry 로짓 (CNN-LSTM과 동일 인터페이스).

설계 요지:
- 각 시간프레임을 Conv2d(stride=patch) 로 패치 토큰화 → N=gh*gw 토큰 (71×120, p=8 → 9×15=135).
- 위치: 패치격자 2D sinusoidal(공간) + 학습가능 시간 임베딩.
- factorized 어텐션: 공간(N 토큰끼리) → 시간(T 토큰끼리) 분리 → full 3D 대비 메모리 절약.
- 헤드: 마지막 시간프레임 패치임베딩 → 각 패치를 p×p 픽셀로 펴서 원해상도 복원.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _2d_sincos_pos_embed(embed_dim: int, gh: int, gw: int) -> torch.Tensor:
    """패치격자(gh×gw)의 2D sinusoidal 위치 인코딩 → (gh*gw, embed_dim).

    embed_dim 절반은 행(위도 방향), 절반은 열(경도 방향)에 할당.
    """
    assert embed_dim % 4 == 0, "embed_dim은 4의 배수여야 함(2축×sin/cos)"
    half = embed_dim // 2

    def _axis(pos: torch.Tensor, dim: int) -> torch.Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000.0 ** omega)
        out = pos.flatten()[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    rows, cols = torch.meshgrid(
        torch.arange(gh, dtype=torch.float32),
        torch.arange(gw, dtype=torch.float32),
        indexing="ij",
    )
    emb_h = _axis(rows, half)   # (N, half)
    emb_w = _axis(cols, half)   # (N, half)
    return torch.cat([emb_h, emb_w], dim=1)   # (N, embed_dim)


class PatchEmbed(nn.Module):
    """프레임별 Conv2d 패치 토큰화."""

    def __init__(self, in_vars: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_vars, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):  # (B*T, V, Hp, Wp)
        x = self.proj(x)                      # (B*T, C, gh, gw)
        _, _, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)      # (B*T, N, C)
        return x, gh, gw


class FactorizedBlock(nn.Module):
    """공간 어텐션 → 시간 어텐션 → MLP (pre-norm residual)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.attn_s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.attn_t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_m = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x, B, T, N):  # x: (B, T, N, C)
        C = x.shape[-1]
        # 공간: 각 시간프레임 내 N 패치끼리
        xs = self.norm_s(x).reshape(B * T, N, C)
        a, _ = self.attn_s(xs, xs, xs)
        x = x + a.reshape(B, T, N, C)
        # 시간: 각 패치 위치에서 T 프레임끼리
        xt = self.norm_t(x).permute(0, 2, 1, 3).reshape(B * N, T, C)
        a, _ = self.attn_t(xt, xt, xt)
        x = x + a.reshape(B, N, T, C).permute(0, 2, 1, 3)
        # MLP
        x = x + self.mlp(self.norm_m(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    def __init__(self, in_vars: int = 4, patch_size: int = 8, embed_dim: int = 256,
                 depth: int = 6, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, time_window: int = 14, **_ignored):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_vars, embed_dim, patch_size)
        # 학습가능 시간 임베딩 (1, T, 1, C)
        self.time_embed = nn.Parameter(torch.zeros(1, time_window, 1, embed_dim))
        nn.init.trunc_normal_(self.time_embed, std=0.02)
        self.blocks = nn.ModuleList([
            FactorizedBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        # 패치 임베딩 → p×p 픽셀 로짓
        self.head = nn.Linear(embed_dim, patch_size * patch_size)
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _pos(self, gh: int, gw: int) -> torch.Tensor:
        """패치격자 2D 위치인코딩(CPU 캐시). device/dtype 변환은 호출부에서.

        device를 캐시 키/값에 섞으면 모델을 .to()로 옮길 때 stale 캐시가 남으므로
        CPU에 보관하고 forward에서 tok에 맞춰 변환한다.
        """
        key = (gh, gw)
        if key not in self._pos_cache:
            self._pos_cache[key] = _2d_sincos_pos_embed(self.embed_dim, gh, gw)
        return self._pos_cache[key]   # (N, C), CPU

    def forward(self, x):  # (B, T, V, H, W)
        B, T, V, H, W = x.shape
        assert T <= self.time_embed.shape[1], \
            f"입력 T={T} > time_window={self.time_embed.shape[1]} (시간 임베딩 범위 초과)"
        ps = self.patch_size
        Hp, Wp = math.ceil(H / ps) * ps, math.ceil(W / ps) * ps
        x = F.pad(x, (0, Wp - W, 0, Hp - H))          # 우/하 패딩(패치 정합)
        tok, gh, gw = self.patch_embed(x.reshape(B * T, V, Hp, Wp))
        N, C = gh * gw, self.embed_dim
        tok = tok.reshape(B, T, N, C)
        pos = self._pos(gh, gw).to(device=tok.device, dtype=tok.dtype)  # device/dtype 정합
        tok = tok + pos[None, None]                           # 공간 위치
        tok = tok + self.time_embed[:, :T]                    # 시간 위치
        for blk in self.blocks:
            tok = blk(tok, B, T, N)
        last = self.norm(tok)[:, -1]                  # (B, N, C) 마지막 시간프레임
        pix = self.head(last)                         # (B, N, ps*ps)
        # 패치격자 → 픽셀맵: (B, gh, gw, ps, ps) → (B, gh*ps, gw*ps)
        pix = pix.reshape(B, gh, gw, ps, ps).permute(0, 1, 3, 2, 4).reshape(B, gh * ps, gw * ps)
        return pix[:, :H, :W]                         # 패딩 제거 → (B, H, W) 로짓
