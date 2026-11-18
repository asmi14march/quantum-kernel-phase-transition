"""Shared benchmarking protocol used by the scripts."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .models import make_model

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"


def split(param, n_train, seed):
    """Stratified by control parameter so every temperature / field appears in train and test."""
    idx = np.arange(len(param))
    return train_test_split(idx, train_size=n_train, stratify=param, random_state=seed)


def tune_and_fit(name, X_tr, y_tr, seed, estimator=None, grid=None, n_jobs=2):
    if estimator is None:
        estimator, grid = make_model(name, seed)
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    gs = GridSearchCV(estimator, grid, cv=cv, n_jobs=n_jobs, refit=True)
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, time.perf_counter() - t0


def score(model, X_te, y_te, param_te, crit, window):
    t0 = time.perf_counter()
    pred = model.predict(X_te)
    t_pred = time.perf_counter() - t0
    near = np.abs(param_te - crit) < window
    if hasattr(model, "decision_function"):
        auc = roc_auc_score(y_te, model.decision_function(X_te))
    else:
        auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    return {
        "accuracy": accuracy_score(y_te, pred),
        "roc_auc": auc,
        "accuracy_near_critical": accuracy_score(y_te[near], pred[near]),
        "predict_time_s": t_pred,
    }


def benchmark(dataset, inputs, y, param, models, seeds, n_train, crit, window, extra=None):
    """inputs: dict model_name -> design matrix. extra: dict model_name -> (estimator, grid)."""
    rows = []
    for seed in seeds:
        tr, te = split(param, n_train, seed)
        for name in models:
            X = inputs[name]
            est, grid = (extra or {}).get(name, (None, None))
            model, best, t_fit = tune_and_fit(name, X[tr], y[tr], seed, est, grid)
            r = score(model, X[te], y[te], param[te], crit, window)
            r.update(dataset=dataset, model=name, seed=seed, fit_time_s=t_fit,
                     best_params=str(best), n_train=len(tr), n_test=len(te))
            rows.append(r)
            print(f"  [{dataset}] seed={seed} {name:18s} acc={r['accuracy']:.4f} "
                  f"near-Tc={r['accuracy_near_critical']:.4f} ({t_fit:.1f}s) {best}", flush=True)
    return pd.DataFrame(rows)


def summarize(df, keys=("dataset", "model")):
    agg = df.groupby(list(keys)).agg(
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        near_critical=("accuracy_near_critical", "mean"),
        near_critical_std=("accuracy_near_critical", "std"),
        roc_auc=("roc_auc", "mean"), fit_time_s=("fit_time_s", "mean"),
    )
    return agg.reset_index()
