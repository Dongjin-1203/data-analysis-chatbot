"""불량 예측 모델 정의와 전처리 (학습 스크립트와 예측 도구가 함께 쓴다).

모델 파일 구성 (models/):
- defect_mlp.pt   : PyTorch state_dict (가중치)
- defect_mlp.json : 입력 열 목록, 자료형, 결측치 대체값, 표준화 평균·표준편차, 판정 임계값, 성능 지표

예측 순서: 입력 열 확인 → 숫자로 변환 → 결측치를 학습 중앙값으로 대체 → 표준화 → MLP → sigmoid 확률
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
WEIGHTS_PATH = MODEL_DIR / "defect_mlp.pt"
META_PATH = MODEL_DIR / "defect_mlp.json"


class DefectMLP(nn.Module):
    """공정 센서 통계값 → 불량(quality_overall=1) 확률을 내는 소형 신경망."""

    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 로짓을 돌려준다
        return self.net(x).squeeze(-1)


@dataclass
class Preprocessor:
    features: list[str]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str]) -> "Preprocessor":
        x = df[features].astype(float)
        medians = x.median()
        filled = x.fillna(medians)
        stds = filled.std().replace(0, 1.0).fillna(1.0)
        return cls(
            features=features,
            medians={k: float(v) for k, v in medians.items()},
            means={k: float(v) for k, v in filled.mean().items()},
            stds={k: float(v) for k, v in stds.items()},
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.features].apply(pd.to_numeric, errors="coerce").astype(float)
        x = x.fillna(pd.Series(self.medians))
        x = (x - pd.Series(self.means)) / pd.Series(self.stds)
        return x.to_numpy(dtype=np.float32)

    def zscores(self, row: pd.Series) -> dict[str, float]:
        """학습 데이터 분포 대비 얼마나 벗어났는지 (결측은 0으로 본다)."""
        out = {}
        for f in self.features:
            v = pd.to_numeric(row.get(f), errors="coerce")
            out[f] = 0.0 if pd.isna(v) else float((v - self.means[f]) / self.stds[f])
        return out


def save(model: DefectMLP, pre: Preprocessor, meta: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    meta = {
        **meta,
        "features": pre.features,
        "dtypes": {f: "float" for f in pre.features},
        "preprocessing": {"impute": "median", "scale": "standard", "medians": pre.medians, "means": pre.means, "stds": pre.stds},
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load(weights_path: Path = WEIGHTS_PATH, meta_path: Path = META_PATH) -> tuple[DefectMLP, Preprocessor, dict]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    p = meta["preprocessing"]
    pre = Preprocessor(features=meta["features"], medians=p["medians"], means=p["means"], stds=p["stds"])
    model = DefectMLP(len(pre.features), tuple(meta["architecture"]["hidden"]), meta["architecture"]["dropout"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model, pre, meta
