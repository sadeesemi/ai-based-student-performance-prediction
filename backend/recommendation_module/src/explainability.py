"""Explainable AI layer.

* global : impurity-based + permutation feature importance
* local  : exact additive tree contributions (the SHAP TreeExplainer decomposition
           for forests - prediction = bias + sum of per-feature contributions)
* local  : LIME-style weighted ridge surrogate for a sampled set of learners
"""
import numpy as np
from scipy import sparse
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


def global_importance(model, feature_names, X_val=None, y_val=None, seed=42):
    out = {"impurity": dict(zip(feature_names, model.feature_importances_.round(6)))}
    if X_val is not None and len(X_val):
        perm = permutation_importance(model, X_val, y_val, n_repeats=3,
                                      random_state=seed, n_jobs=-1)
        out["permutation"] = dict(zip(feature_names, perm.importances_mean.round(6)))
    return out


def tree_contributions(forest, X, n_features):
    """Exact additive attribution: pred = bias + contributions.sum(axis=1)."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    total = np.zeros((X.shape[0], n_features))
    bias = 0.0
    for est in forest.estimators_:
        t = est.tree_
        values = t.value[:, 0, 0]
        left, right, feat = t.children_left, t.children_right, t.feature
        parent = np.full(t.node_count, -1, dtype=np.int64)
        pfeat = np.zeros(t.node_count, dtype=np.int64)
        internal = np.where(left != -1)[0]
        parent[left[internal]] = internal
        parent[right[internal]] = internal
        pfeat[left[internal]] = feat[internal]
        pfeat[right[internal]] = feat[internal]
        has_parent = parent >= 0
        delta = np.zeros(t.node_count)
        delta[has_parent] = values[has_parent] - values[parent[has_parent]]
        P = est.decision_path(X).astype(np.float64).multiply(delta[None, :]).tocsr()
        rows = np.where(has_parent)[0]
        onehot = sparse.csr_matrix((np.ones(len(rows)), (rows, pfeat[rows])),
                                   shape=(t.node_count, n_features))
        total += np.asarray((P @ onehot).todense())
        bias += values[0]
    n = len(forest.estimators_)
    return total / n, bias / n


def lime_explain(model, x, X_ref, feature_names, n_samples=400, seed=42):
    """Local surrogate: perturb around x, weight by proximity, fit ridge."""
    rng = np.random.default_rng(seed)
    scale = X_ref.std(axis=0) + 1e-6
    Z = x[None, :] + rng.normal(0, 0.5, size=(n_samples, len(x))) * scale
    Z = np.clip(Z, X_ref.min(axis=0), X_ref.max(axis=0)).astype(np.float32)
    yz = model.predict(Z)
    d = np.linalg.norm((Z - x[None, :]) / scale, axis=1)
    w = np.exp(-(d ** 2) / (2 * (0.75 * np.sqrt(len(x))) ** 2))
    ridge = Ridge(alpha=1.0).fit(Z, yz, sample_weight=w)
    return dict(zip(feature_names, ridge.coef_.round(6)))


def top_contributions(contrib_row, feature_names, pretty, k=8):
    order = np.argsort(-np.abs(contrib_row))[:k]
    return [{"feature": feature_names[i], "label": pretty.get(feature_names[i], feature_names[i]),
             "impact": round(float(contrib_row[i]), 5)} for i in order]
