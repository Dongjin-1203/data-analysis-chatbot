from pathlib import Path

import pandas as pd
import pytest

from app import defect_model
from app.predictor import (
    PredictionError,
    check_columns,
    model_info,
    predict_record,
    predict_values,
    required_features,
    standalone_code_example,
)

FIXTURES = Path(__file__).parent / "fixtures"
ALCU = pd.read_csv(FIXTURES / "alcu_sample.csv", low_memory=False)
INSPECTION = pd.read_csv(FIXTURES / "sample_inspection.csv")


def test_model_files_and_metadata():
    assert defect_model.WEIGHTS_PATH.exists() and defect_model.META_PATH.exists()
    info = model_info()
    assert info["label"] == "quality_overall"
    assert 0 < info["threshold"] < 1
    assert info["n_features"] == len(required_features()) > 50
    # 계측 결과·라벨 열은 입력에 들어가면 안 된다 (데이터 누수 방지)
    assert not any(f.startswith("RAW_VALUE") or f.startswith("quality") for f in required_features())
    assert "mean_thickness" not in required_features()


def test_predict_by_wafer_id_and_row_index():
    by_id = predict_record(ALCU, wafer_id="V612744112")
    row = int(ALCU.index[ALCU["wafer_id"] == "V612744112"][0])
    by_row = predict_record(ALCU, row_index=row)
    assert by_id["score"] == by_row["score"]
    assert by_id["record"]["wafer_id"] == "V612744112"
    assert by_id["actual_label"] == 1
    assert by_id["label"] == 1 and by_id["prediction"] == "불량 의심"
    assert len(by_id["top_deviations"]) == 5
    assert {"feature", "value", "train_mean", "z"} <= set(by_id["top_deviations"][0])


def test_good_wafer_predicted_good():
    result = predict_record(ALCU, row_index=0)
    assert result["actual_label"] == 0 and result["label"] == 0


def test_predictions_are_deterministic():
    assert predict_record(ALCU, row_index=3)["score"] == predict_record(ALCU, row_index=3)["score"]


def test_unsupported_data_missing_columns():
    assert len(check_columns(INSPECTION)) == len(required_features())
    with pytest.raises(PredictionError) as e:
        predict_record(INSPECTION, row_index=0)
    assert e.value.kind == "unsupported_input"
    assert "입력 열" in e.value.message and "없습니다" in e.value.message


def test_non_numeric_value_rejected():
    broken = ALCU.copy()
    feature = required_features()[0]
    broken[feature] = broken[feature].astype(object)
    broken.loc[0, feature] = "N/A값"
    with pytest.raises(PredictionError) as e:
        predict_record(broken, row_index=0)
    assert e.value.kind == "unsupported_input" and feature in e.value.message


def test_too_many_missing_rejected():
    sparse = ALCU.copy()
    sparse.loc[0, required_features()[:60]] = None
    with pytest.raises(PredictionError) as e:
        predict_record(sparse, row_index=0)
    assert "신뢰할 수 없습니다" in e.value.message


@pytest.mark.parametrize(
    "kwargs, text",
    [({}, "지정해 주세요"), ({"row_index": 999}, "범위"), ({"wafer_id": "NOPE"}, "없습니다")],
)
def test_bad_requests(kwargs, text):
    with pytest.raises(PredictionError) as e:
        predict_record(ALCU, **kwargs)
    assert e.value.kind == "bad_request" and text in e.value.message


def test_predict_values():
    row = ALCU.iloc[0]
    values = {f: float(row[f]) for f in required_features() if pd.notna(row[f])}
    assert predict_values(values)["score"] == predict_record(ALCU, row_index=0)["score"]
    with pytest.raises(PredictionError):
        predict_values({"POWER_TARGET_DEP1_A": 1.0})


def test_standalone_code_example_runs(tmp_path, monkeypatch):
    """별도 환경 안내 코드가 실제로 실행되고 도구와 같은 점수를 낸다."""
    import shutil
    import subprocess
    import sys

    (tmp_path / "models").mkdir()
    shutil.copy(defect_model.WEIGHTS_PATH, tmp_path / "models")
    shutil.copy(defect_model.META_PATH, tmp_path / "models")
    ALCU.head(3).to_csv(tmp_path / "내_데이터.csv", index=False)
    code = standalone_code_example() + "\nprint('SCORE0', round(float(score[0]), 4))\n"
    (tmp_path / "run.py").write_text(code, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "run.py"], cwd=tmp_path, capture_output=True, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    )
    out = proc.stdout.decode("utf-8")
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert f"SCORE0 {predict_record(ALCU, row_index=0)['score']}" in out
