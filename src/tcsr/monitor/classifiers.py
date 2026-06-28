"""sklearn-compatible monitor classifiers: LogReg, RF, XGB."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LogRegMonitor:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str = "balanced"):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight)),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogRegMonitor":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        coef = self.pipeline.named_steps["clf"].coef_[0]
        return dict(zip(feature_names, coef.tolist()))


class RFMonitor:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth=None,
        min_samples_leaf: int = 5,
        class_weight: str = "balanced",
        n_jobs: int = -1,
    ):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFMonitor":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.clf.predict_proba(X)
        if proba.shape[1] == 1:
            return np.zeros(len(X), dtype=np.float32)
        return proba[:, 1]

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        return dict(zip(feature_names, self.clf.feature_importances_.tolist()))


class XGBMonitor:
    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight=None,
        eval_metric: str = "aucpr",
        early_stopping_rounds: int = 30,
        random_state: int = 0,
    ):
        from xgboost import XGBClassifier

        self._scale_pos_weight = scale_pos_weight
        self._early = early_stopping_rounds
        self._eval_metric = eval_metric
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            use_label_encoder=False,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "XGBMonitor":
        spw = self._scale_pos_weight
        if spw is None:
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            spw = n_neg / max(n_pos, 1)
        self.clf.set_params(scale_pos_weight=spw)

        eval_set = [(X_val, y_val)] if X_val is not None else [(X, y)]
        self.clf.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        scores = self.clf.feature_importances_
        return dict(zip(feature_names, scores.tolist()))


def save_monitor(monitor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(monitor, f)


def load_monitor(path: Path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)
