"""불량 예측 도구의 본체: 제공된 가중치를 CPU에서 불러와 검사 건 한 건을 예측한다.

지원 범위
- 모델: models/defect_mlp.pt + defect_mlp.json (PVD AlCu 공정 센서 → quality_overall 불량 여부)
- 입력: 메타데이터에 적힌 입력 열(features)을 모두 가진 표 데이터의 한 행, 또는 열 이름→값 딕셔너리
지원하지 않는 입력(필수 열 누락, 숫자가 아닌 값, 결측이 너무 많은 행, 모델 없음)은
PredictionError 로 이유를 알리고, 별도 환경에서 실행할 코드 예시를 함께 준다.
"""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from app import defect_model

MAX_MISSING_RATIO = 0.5  # 한 행에서 입력 열의 절반 넘게 비어 있으면 예측하지 않는다
TOP_DEVIATIONS = 5


def _sig(x: float, digits: int = 4) -> float:
    """아주 작은 값(예: 1e-7 단위 압력)도 0 으로 보이지 않도록 유효숫자로 반올림한다."""
    return float(f"{x:.{digits}g}")


class PredictionError(Exception):
    def __init__(self, kind: str, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.kind = kind  # no_model | unsupported_input | bad_request
        self.message = message
        self.details = details or {}


@lru_cache
def _load():
    if not defect_model.WEIGHTS_PATH.exists() or not defect_model.META_PATH.exists():
        raise PredictionError(
            "no_model",
            "예측 모델 파일이 없습니다. `uv run python scripts/train_defect_model.py` 로 먼저 학습해 주세요.",
        )
    return defect_model.load()


def model_info() -> dict:
    _, pre, meta = _load()
    test = meta.get("metrics", {}).get("test", {})
    return {
        "name": meta.get("model"),
        "label": meta.get("label"),
        "label_meaning": meta.get("label_meaning"),
        "threshold": meta.get("threshold"),
        "n_features": len(pre.features),
        "test_roc_auc": test.get("roc_auc"),
        "test_f1": test.get("f1"),
        "test_precision": test.get("precision"),
        "test_recall": test.get("recall"),
        "note": "점수는 보정된 확률이 아니므로 임계값과 비교해 판정합니다.",
    }


def required_features() -> list[str]:
    return list(_load()[1].features)


def check_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력 열 중 데이터에 없는 열."""
    return [f for f in required_features() if f not in df.columns]


def _select_row(df: pd.DataFrame, row_index: int | None, wafer_id: str | None) -> tuple[int, pd.Series]:
    if wafer_id is not None:
        if "wafer_id" not in df.columns:
            raise PredictionError("bad_request", "데이터에 wafer_id 열이 없습니다. row_index 로 행을 지정해 주세요.")
        matches = df.index[df["wafer_id"].astype(str) == str(wafer_id)].tolist()
        if not matches:
            raise PredictionError("bad_request", f"wafer_id '{wafer_id}' 인 행이 없습니다.")
        if len(matches) > 1:
            raise PredictionError(
                "bad_request",
                f"wafer_id '{wafer_id}' 인 행이 {len(matches)}개입니다. row_index 로 하나를 지정해 주세요.",
                {"rows": [int(df.index.get_loc(m)) for m in matches[:10]]},
            )
        pos = int(df.index.get_loc(matches[0]))
        return pos, df.iloc[pos]
    if row_index is None:
        raise PredictionError("bad_request", "예측할 검사 건을 row_index(0부터 시작) 또는 wafer_id 로 지정해 주세요.")
    if not 0 <= row_index < len(df):
        raise PredictionError("bad_request", f"row_index 는 0 ~ {len(df) - 1} 범위여야 합니다. (입력: {row_index})")
    return row_index, df.iloc[row_index]


def _validate_row(row: pd.Series, features: list[str]) -> int:
    values = pd.to_numeric(row[features], errors="coerce")
    non_numeric = [f for f in features if pd.notna(row[f]) and pd.isna(values[f])]
    if non_numeric:
        raise PredictionError(
            "unsupported_input",
            f"숫자가 아닌 값이 들어 있는 입력 열이 {len(non_numeric)}개 있습니다: {', '.join(non_numeric[:10])}",
            {"non_numeric": non_numeric},
        )
    n_missing = int(values.isna().sum())
    if n_missing > len(features) * MAX_MISSING_RATIO:
        raise PredictionError(
            "unsupported_input",
            f"입력 열 {len(features)}개 중 {n_missing}개가 비어 있어 예측을 신뢰할 수 없습니다 "
            f"(허용 한도 {int(MAX_MISSING_RATIO * 100)}%).",
            {"missing_count": n_missing},
        )
    return n_missing


def _predict(row: pd.Series) -> dict:
    model, pre, meta = _load()
    n_missing = _validate_row(row, pre.features)
    x = pre.transform(pd.DataFrame([row]))
    with torch.no_grad():
        score = float(torch.sigmoid(model(torch.from_numpy(x)))[0])
    threshold = float(meta["threshold"])
    label = int(score >= threshold)

    z = pre.zscores(row)
    top = sorted(z.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_DEVIATIONS]
    deviations = [
        {
            "feature": f,
            "value": None if pd.isna(pd.to_numeric(row.get(f), errors="coerce")) else _sig(float(row[f])),
            "train_mean": _sig(pre.means[f]),
            "z": round(v, 2),
        }
        for f, v in top
    ]
    return {
        "score": round(score, 4),
        "threshold": threshold,
        "label": label,
        "prediction": "불량 의심" if label else "양품 예상",
        "imputed_features": n_missing,
        "top_deviations": deviations,
    }


def predict_record(df: pd.DataFrame, row_index: int | None = None, wafer_id: str | None = None) -> dict:
    missing = check_columns(df)
    if missing:
        raise PredictionError(
            "unsupported_input",
            f"이 데이터에는 모델 입력 열 {len(required_features())}개 중 {len(missing)}개가 없습니다. "
            f"(예: {', '.join(missing[:8])}) 학습에 쓴 PVD AlCu 공정 데이터와 같은 형식이어야 합니다.",
            {"missing_columns": missing},
        )
    pos, row = _select_row(df, row_index, wafer_id)
    result = _predict(row)
    record = {"row_index": pos}
    for col in ("wafer_id", "lot", "equipment", "machine", "slot", "runstart"):
        if col in df.columns and pd.notna(row.get(col)):
            record[col] = row[col].item() if isinstance(row[col], np.generic) else row[col]
    actual = None
    if "quality_overall" in df.columns and pd.notna(row.get("quality_overall")):
        actual = int(row["quality_overall"])
    return {"record": record, **result, "actual_label": actual, "model": model_info()}


def predict_values(values: dict) -> dict:
    features = required_features()
    missing = [f for f in features if f not in values]
    if len(missing) > len(features) * MAX_MISSING_RATIO:
        raise PredictionError(
            "unsupported_input",
            f"입력 열 {len(features)}개 중 {len(missing)}개가 없어 예측할 수 없습니다. 업로드한 데이터의 행을 지정하는 방식을 권장합니다.",
            {"missing_columns": missing},
        )
    row = pd.Series({f: values.get(f, np.nan) for f in features})
    return {"record": {"source": "직접 입력한 값"}, **_predict(row), "actual_label": None, "model": model_info()}


def standalone_code_example() -> str:
    """챗봇이 처리할 수 없는 입력일 때 사용자가 별도 환경에서 실행해 볼 수 있는 코드."""
    return '''# 별도 환경에서 불량 예측 모델 실행하기 (CPU)
# 준비: pip install torch pandas  /  models/defect_mlp.pt, models/defect_mlp.json 복사
import json
import pandas as pd
import torch
from torch import nn

meta = json.load(open("models/defect_mlp.json", encoding="utf-8"))
features = meta["features"]
prep = meta["preprocessing"]

class DefectMLP(nn.Module):
    def __init__(self, n, hidden, dropout):
        super().__init__()
        layers, prev = [], n
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = DefectMLP(len(features), meta["architecture"]["hidden"], meta["architecture"]["dropout"])
model.load_state_dict(torch.load("models/defect_mlp.pt", map_location="cpu", weights_only=True))
model.eval()

df = pd.read_csv("내_데이터.csv")              # 입력 열 이름을 features 와 맞춰야 합니다
missing = [f for f in features if f not in df.columns]
print("없는 입력 열:", missing)               # 없는 열은 학습 중앙값으로 채워지므로 결과를 신뢰하기 어렵습니다
for f in missing:
    df[f] = float("nan")

x = df[features].apply(pd.to_numeric, errors="coerce")
x = x.fillna(pd.Series(prep["medians"]))
x = (x - pd.Series(prep["means"])) / pd.Series(prep["stds"])
with torch.no_grad():
    score = torch.sigmoid(model(torch.tensor(x.values, dtype=torch.float32))).numpy()
df["defect_score"] = score
df["defect_pred"] = (score >= meta["threshold"]).astype(int)
print(df[["defect_score", "defect_pred"]].head())
'''
