"""공통 유틸: 설정 로드, 시드 고정."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """YAML 설정 파일 로드. 상대경로는 repo 루트 기준."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """재현성을 위한 전역 시드 고정 (numpy/random/torch)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_path(rel: str | Path) -> Path:
    """repo 루트 기준 절대경로 반환."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p
