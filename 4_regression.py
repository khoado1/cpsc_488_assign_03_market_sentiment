#!/usr/bin/env python3
"""Train PyTorch MLP regressors for MIV prediction from vectorized news text."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets"


class MLPRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_dims: Iterable[int] = (128, 64)) -> None:
        super().__init__()
        dims = [in_features, *hidden_dims, 1]
        blocks: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            blocks.append(nn.Linear(dims[idx], dims[idx + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.Dropout(p=0.1))
        blocks.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def parse_vector(raw_value: object) -> np.ndarray:
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return np.zeros(0, dtype=np.float32)
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        return np.asarray(raw_value, dtype=np.float32).ravel()
    if not isinstance(raw_value, str):
        return np.asarray([float(raw_value)], dtype=np.float32)

    text = raw_value.strip()
    if text == "" or text.lower() in {"nan", "none", "null", "[]"}:
        return np.zeros(0, dtype=np.float32)

    cleaned = text.replace("'", '"')
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            matches = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
            value = [float(token) for token in matches]

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32).ravel()
    if isinstance(value, str):
        numeric = re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        return np.asarray([float(v) for v in numeric], dtype=np.float32)
    return np.asarray([float(value)], dtype=np.float32)


def pad_feature_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("No vector rows available to train on.")
    max_len = max(arr.size for arr in vectors)
    matrix = np.zeros((len(vectors), max_len), dtype=np.float32)
    for idx, arr in enumerate(vectors):
        if arr.size:
            matrix[idx, : arr.size] = arr[:max_len]
    return matrix


def load_vectorized_dataset(path: str | Path, *, target_col: str = "MIV") -> pd.DataFrame:
    dataset = pd.read_csv(path)
    required = {"date", "symbol", "news_vector", target_col}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
    dataset = dataset.copy()
    dataset["symbol"] = dataset["symbol"].astype(str).str.strip().str.upper()
    dataset[target_col] = pd.to_numeric(dataset[target_col], errors="coerce")
    if dataset[target_col].isna().any():
        bad = int(dataset[target_col].isna().sum())
        raise ValueError(f"{path}: {bad} invalid/blank MIV values found")
    dataset["vector"] = dataset["news_vector"].map(parse_vector)
    dataset = dataset[["date", "symbol", "vector", target_col]].copy()
    if dataset.empty:
        raise ValueError(f"{path}: dataset is empty")
    return dataset


def fit_regressor(train_X: np.ndarray, train_y: np.ndarray, *, seed: int = 42, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPRegressor(in_features=train_X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model


def predict_regression(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32))
    return preds.cpu().numpy()


def evaluate_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, *, seed: int = 42, epochs: int = 60, batch_size: int = 32, lr: float = 1e-3) -> dict[str, object]:
    train_X = pad_feature_matrix(train_df["vector"].tolist())
    test_X = pad_feature_matrix(test_df["vector"].tolist())
    max_dim = max(train_X.shape[1], test_X.shape[1])
    if train_X.shape[1] < max_dim:
        pad = np.zeros((train_X.shape[0], max_dim - train_X.shape[1]), dtype=np.float32)
        train_X = np.hstack([train_X, pad])
    if test_X.shape[1] < max_dim:
        pad = np.zeros((test_X.shape[0], max_dim - test_X.shape[1]), dtype=np.float32)
        test_X = np.hstack([test_X, pad])

    model = fit_regressor(train_X, train_df["MIV"].to_numpy(), seed=seed, epochs=epochs, batch_size=batch_size, lr=lr)
    pred = predict_regression(model, test_X).reshape(-1)
    actual = test_df["MIV"].to_numpy(dtype=float).reshape(-1)

    residuals = actual - pred
    rmse = math.sqrt(np.mean(residuals ** 2))
    denom = np.sum((actual - actual.mean()) ** 2)
    r2 = 1.0 - (np.sum(residuals ** 2) / denom) if denom else 1.0

    company_rows = []
    for symbol, group in pd.DataFrame({"symbol": test_df["symbol"], "actual": actual, "predicted": pred}).groupby("symbol"):
        y = group["actual"].to_numpy(dtype=float)
        y_hat = group["predicted"].to_numpy(dtype=float)
        residual = y - y_hat
        denom_symbol = np.sum((y - y.mean()) ** 2)
        r2_symbol = 1.0 - (np.sum(residual ** 2) / denom_symbol) if denom_symbol else 1.0
        rmse_symbol = math.sqrt(np.mean(residual ** 2))
        company_rows.append({"symbol": symbol, "observations": len(group), "rmse": rmse_symbol, "r2": r2_symbol})
    company_stats = pd.DataFrame(company_rows).sort_values(["observations", "rmse"], ascending=[False, True]).reset_index(drop=True)

    return {
        "rmse": rmse,
        "r2": r2,
        "company_stats": company_stats,
        "model": model,
    }


def resolve_default_paths(vectorizer: str | None = None) -> tuple[Path, Path]:
    candidates = {
        "dtm": (DATA_DIR / "news_dtm_train.csv", DATA_DIR / "news_dtm_test.csv"),
        "tfidf": (DATA_DIR / "news_tfidf_train.csv", DATA_DIR / "news_tfidf_test.csv"),
    }
    sample_candidates = {
        "dtm": (DATA_DIR / "sample_news_dtm_train.csv", DATA_DIR / "sample_news_dtm_test.csv"),
        "tfidf": (DATA_DIR / "sample_news_tfidf_train.csv", DATA_DIR / "sample_news_tfidf_test.csv"),
    }
    if vectorizer is not None:
        if vectorizer not in candidates:
            raise ValueError(f"Unsupported vectorizer: {vectorizer}. Choose from {sorted(candidates)}")
        train_path, test_path = candidates[vectorizer]
        if train_path.exists() and test_path.exists():
            return train_path, test_path
        sample_train, sample_test = sample_candidates[vectorizer]
        if sample_train.exists() and sample_test.exists():
            return sample_train, sample_test
        raise FileNotFoundError(f"Could not find {vectorizer} train/test CSVs in datasets/.")

    for key in ("dtm", "tfidf"):
        train_p, test_p = candidates[key]
        if train_p.exists() and test_p.exists():
            return train_p, test_p
        sample_train, sample_test = sample_candidates[key]
        if sample_train.exists() and sample_test.exists():
            return sample_train, sample_test
    raise FileNotFoundError("Could not find default train/test CSVs in datasets/.")


def run_vectorizer(vectorizer: str, *, epochs: int, batch_size: int, lr: float, seed: int) -> dict[str, object]:
    train_path, test_path = resolve_default_paths(vectorizer)
    train_df = load_vectorized_dataset(train_path)
    test_df = load_vectorized_dataset(test_path)
    metrics = evaluate_regressor(train_df, test_df, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr)
    metrics["vectorizer"] = vectorizer
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a PyTorch MLP regressor on DTM or TF-IDF market news vectors.")
    parser.add_argument("--vectorizer", choices=["dtm", "tfidf", "both"], default="both", help="Which vectorizer to evaluate")
    parser.add_argument("--train", type=str, default=None, help="Optional training CSV path. Used with --test for a custom dataset")
    parser.add_argument("--test", type=str, default=None, help="Optional testing CSV path. Used with --train for a custom dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    try:
        if args.train and args.test:
            train_df = load_vectorized_dataset(args.train)
            test_df = load_vectorized_dataset(args.test)
            metrics = evaluate_regressor(train_df, test_df, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            result_map = {"custom": metrics}
        else:
            selected = [args.vectorizer] if args.vectorizer != "both" else ["dtm", "tfidf"]
            result_map = {name: run_vectorizer(name, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed) for name in selected}

        for key, metrics in result_map.items():
            print(f"\n=== {key.upper()} RESULTS ===")
            print(f"RMSE: {metrics['rmse']:.4f}")
            print(f"R²: {metrics['r2']:.4f}")
            topk = metrics["company_stats"].head(args.top_k)
            print(f"\nTop {len(topk)} companies by test observations:")
            print(topk[["symbol", "observations", "rmse", "r2"]].to_string(index=False))

        if len(result_map) > 1:
            best_key, best_metrics = min(result_map.items(), key=lambda item: item[1]["rmse"])
            print(f"\nBest-performing vectorizer by RMSE: {best_key.upper()} (RMSE={best_metrics['rmse']:.4f})")

        return 0
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
