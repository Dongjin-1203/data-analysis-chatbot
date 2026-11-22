"""PVD AlCu 공정 데이터로 불량(quality_overall) 예측 소형 신경망을 학습한다.

사용법:
    uv run python scripts/train_defect_model.py --data data/alcu_v2.csv

결과: models/defect_mlp.pt (가중치), models/defect_mlp.json (입력 열·전처리·임계값·성능)

데이터: Mević, Laber, Krivić, "Advanced Process Control and Statistical Process Control Data for
Thickness Prediction of AlCu and WTi Metal Layer in Semiconductor Manufacturing", Zenodo,
doi:10.5281/zenodo.23442538 (CC BY 4.0)
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.defect_model import DefectMLP, Preprocessor, save  # noqa: E402

LABEL = "quality_overall"
META_COLUMNS = {"bat_id", "wafer_id", "lot", "equipment", "machine", "batchstart", "batchend", "runstart", "runend", "slot"}
QUALITY_COLUMNS = {"stable_state", "quality_thickness", "quality_resistance", "quality_resistivity", "quality_overall"}


def is_output_column(name: str) -> bool:
    """계측 결과(SPC) 열은 예측 시점에 알 수 없으므로 입력에서 뺀다."""
    return name.startswith("RAW_VALUE") or name in {"mean_thickness", "mean_resistance"}


def select_features(df: pd.DataFrame, max_missing: float) -> tuple[list[str], dict]:
    candidates = [
        c for c in df.columns
        if c not in META_COLUMNS and c not in QUALITY_COLUMNS and not is_output_column(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    missing = df[candidates].isna().mean()
    too_missing = sorted(missing[missing > max_missing].index)
    kept = [c for c in candidates if c not in too_missing]
    constant = sorted(c for c in kept if df[c].nunique(dropna=True) <= 1)
    features = [c for c in kept if c not in constant]
    report = {"candidates": len(candidates), "dropped_missing": too_missing, "dropped_constant": constant}
    return features, report


def stratified_split(y: np.ndarray, seed: int, ratios=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        n_train = int(len(idx) * ratios[0])
        n_val = int(len(idx) * ratios[1])
        parts[0] += list(idx[:n_train])
        parts[1] += list(idx[n_train:n_train + n_val])
        parts[2] += list(idx[n_train + n_val:])
    return [np.array(sorted(p)) for p in parts]


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "roc_auc": round(roc_auc(y, prob), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def best_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 91)
    scores = [binary_metrics(y, prob, t)["f1"] for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def predict(model: DefectMLP, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x))).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/alcu_v2.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-missing", type=float, default=0.3, help="결측률이 이보다 큰 열은 입력에서 제외")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data, low_memory=False)
    df = df[df[LABEL].isin([0, 1])].reset_index(drop=True)
    y = df[LABEL].astype(int).to_numpy()
    features, report = select_features(df, args.max_missing)
    print(f"데이터 {len(df)}행, 불량 비율 {y.mean():.2%}")
    print(f"입력 열 {len(features)}개 (후보 {report['candidates']}개, 결측 과다 제외 {len(report['dropped_missing'])}개, "
          f"상수 열 제외 {len(report['dropped_constant'])}개)")

    train_idx, val_idx, test_idx = stratified_split(y, args.seed)
    pre = Preprocessor.fit(df.iloc[train_idx], features)  # 전처리 통계는 학습 데이터로만 계산
    x_all = pre.transform(df)
    x_train, x_val, x_test = x_all[train_idx], x_all[val_idx], x_all[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    model = DefectMLP(len(features), tuple(args.hidden), args.dropout)
    pos_weight = torch.tensor((y_train == 0).sum() / max((y_train == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 불량이 적으므로 불량 샘플에 가중치
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    best_auc, best_state, best_epoch, waited = -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), args.batch_size):
            batch = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        val_auc = roc_auc(y_val, predict(model, x_val))
        if val_auc > best_auc:
            best_auc, best_epoch, waited = val_auc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val AUC {val_auc:.4f}")
        if waited >= args.patience:
            print(f"조기 종료: epoch {epoch} (최고 val AUC {best_auc:.4f} @ epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    threshold = best_threshold(y_val, predict(model, x_val))
    test = binary_metrics(y_test, predict(model, x_test), threshold)
    baseline_acc = round(float((y_test == 0).mean()), 4)
    print(f"판정 임계값 {threshold} (검증 F1 최대)")
    print(f"테스트: AUC {test['roc_auc']}, 정밀도 {test['precision']}, 재현율 {test['recall']}, F1 {test['f1']}, "
          f"정확도 {test['accuracy']} (전부 양품으로 찍을 때 정확도 {baseline_acc})")
    print(f"혼동행렬: {test['confusion']}")

    save(model, pre, {
        "model": "DefectMLP",
        "framework": f"torch {torch.__version__}",
        "task": "binary_classification",
        "label": LABEL,
        "label_meaning": {"0": "양품 (두께·면저항·비저항 모두 Good)", "1": "불량 (하나 이상 Bad)"},
        "threshold": threshold,
        "architecture": {"hidden": args.hidden, "dropout": args.dropout},
        "excluded_columns": {
            "metadata": sorted(META_COLUMNS),
            "quality_flags": sorted(QUALITY_COLUMNS),
            "metrology_outputs": "RAW_VALUE_*, mean_thickness, mean_resistance (예측 시점에 알 수 없는 계측 결과)",
            "too_many_missing": report["dropped_missing"],
            "constant": report["dropped_constant"],
        },
        "training": {
            "data": Path(args.data).name,
            "source": "Zenodo doi:10.5281/zenodo.23442538 (CC BY 4.0)",
            "rows": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
            "positive_rate": round(float(y.mean()), 4),
            "best_epoch": best_epoch,
            "seed": args.seed,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "metrics": {"test": test, "test_baseline_accuracy_all_good": baseline_acc, "val_roc_auc": round(best_auc, 4)},
    })
    print("저장: models/defect_mlp.pt, models/defect_mlp.json")


if __name__ == "__main__":
    main()
