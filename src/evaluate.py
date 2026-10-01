"""
Metrics follow the paper's notebook: per-fold weighted precision/recall/F1 on
train and validation folds, averaged over folds; per-class precision averaged
over folds. Everyone uses data/folds.pkl so results are directly comparable.

Usage (from a notebook in notebooks/):
    import sys; sys.path.append('../src')
    from evaluate import load_data, evaluate_model, save_results
    d, folds = load_data()
    res = evaluate_model(lambda: SVC(kernel='linear', decision_function_shape='ovo'),
                         d['X_onehot'], d['y'], folds)
    save_results('svm_linear', res)

Custom/precomputed kernels: use evaluate_fn with your own fit_predict(tr, te).
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def load_data(data_dir=DATA_DIR):
    """Returns (npz dict with X_onehot, X_ordinal, y [1..5], y0 [0..4]), folds [(train_idx, test_idx)]."""
    d = np.load(os.path.join(data_dir, 'processed.npz'))
    with open(os.path.join(data_dir, 'folds.pkl'), 'rb') as f:
        folds = pickle.load(f)
    return {k: d[k] for k in d.files}, folds


def _scores(y_true, y_pred, labels):
    return (
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
    )


def evaluate_fn(fit_predict, y, folds):
    """fit_predict(train_idx, test_idx) -> (pred_train, pred_test), predictions in the same label space as y."""
    y = np.asarray(y)
    labels = np.unique(y)
    tr_s, te_s, tr_c, te_c = [], [], [], []
    for tr, te in folds:
        pred_tr, pred_te = fit_predict(tr, te)
        p, r, f, c = _scores(y[tr], pred_tr, labels); tr_s.append((p, r, f)); tr_c.append(c)
        p, r, f, c = _scores(y[te], pred_te, labels); te_s.append((p, r, f)); te_c.append(c)
    tr_m, te_m = np.mean(tr_s, axis=0), np.mean(te_s, axis=0)
    return {
        'train': dict(zip(['precision', 'recall', 'f1'], tr_m)),
        'valid': dict(zip(['precision', 'recall', 'f1'], te_m)),
        'train_class_precision': np.mean(tr_c, axis=0),
        'valid_class_precision': np.mean(te_c, axis=0),
        'labels': labels,
    }


def evaluate_model(make_model, X, y, folds):
    """make_model() returns a fresh estimator with fit/predict. X is a feature matrix indexed by fold indices."""
    X, y = np.asarray(X), np.asarray(y)

    def fit_predict(tr, te):
        m = make_model().fit(X[tr], y[tr])
        return m.predict(X[tr]), m.predict(X[te])

    return evaluate_fn(fit_predict, y, folds)


def to_row(name, res, **extra):
    """Flatten a result dict into one CSV row. `extra` adds columns (e.g. num_trees=100)."""
    row = {'model': name, **extra}
    for split in ('train', 'valid'):
        for k in ('precision', 'recall', 'f1'):
            row[f'{split}_{k}'] = round(float(res[split][k]), 4)
        for i, v in enumerate(res[f'{split}_class_precision'], start=1):
            row[f'{split}_bin{i}_precision'] = round(float(v), 4)
    return row


def save_results(name, res_or_rows, results_dir=RESULTS_DIR, **extra):
    """Write results/<name>.csv. Pass one result dict, or a list of rows from to_row() (e.g. a num_trees sweep)."""
    os.makedirs(results_dir, exist_ok=True)
    rows = res_or_rows if isinstance(res_or_rows, list) else [to_row(name, res_or_rows, **extra)]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, f'{name}.csv'), index=False)
    return df
