"""Evaluation: ranking quality, regression quality, risk-band breakdown and
strategy ablation for the hybrid recommender."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from scipy.stats import spearmanr


def _dcg(gains):
    return float(np.sum(gains / np.log2(np.arange(2, len(gains) + 2))))


def ranking_metrics(scores, rel_sets, ks=(1, 3, 5, 10)):
    out = {f"precision@{k}": [] for k in ks}
    out.update({f"recall@{k}": [] for k in ks})
    out.update({f"ndcg@{k}": [] for k in ks})
    out.update({f"map@{k}": [] for k in ks})
    hits, ranks = [], []
    order_all = np.argsort(-scores, axis=1)
    for i, rel in enumerate(rel_sets):
        order = order_all[i]
        if not rel:
            continue
        rel_flags = np.array([1.0 if j in rel else 0.0 for j in order])
        first = np.argmax(rel_flags) if rel_flags.any() else -1
        hits.append(1.0 if rel_flags[:10].any() else 0.0)
        ranks.append(1.0 / (first + 1) if first >= 0 else 0.0)
        for k in ks:
            top = rel_flags[:k]
            out[f"precision@{k}"].append(top.sum() / k)
            out[f"recall@{k}"].append(top.sum() / len(rel))
            ideal = _dcg(np.ones(min(k, len(rel))))
            out[f"ndcg@{k}"].append(_dcg(top) / ideal if ideal > 0 else 0.0)
            cum = np.cumsum(top)
            prec_at_hit = [(cum[j] / (j + 1)) for j in range(k) if top[j] > 0]
            out[f"map@{k}"].append(float(np.mean(prec_at_hit)) if prec_at_hit else 0.0)
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["hit_rate@10"] = float(np.mean(hits))
    res["mrr"] = float(np.mean(ranks))
    return {k: round(v, 4) for k, v in res.items()}


def catalogue_metrics(scores, resources, top_k=6):
    order = np.argsort(-scores, axis=1)[:, :top_k]
    counts = np.bincount(order.reshape(-1), minlength=scores.shape[1])
    lessons = resources["Lesson_Name"].to_numpy()
    diversity = np.mean([len(set(lessons[row])) / top_k for row in order])
    p = counts / counts.sum()
    nz = p[p > 0]
    return {
        "catalogue_coverage": round(float((counts > 0).mean()), 4),
        "intra_list_lesson_diversity": round(float(diversity), 4),
        "distribution_entropy": round(float(-(nz * np.log2(nz)).sum()), 4),
        "gini": round(float(1 - np.sum(p ** 2)), 4),
        "resource_hit_counts": counts.tolist(),
    }


def regression_metrics(y_true, y_pred):
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "spearman": round(float(spearmanr(y_true, y_pred).statistic), 4),
    }


def classification_metrics(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    if y_proba is not None:
        out["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return out


def band_breakdown(scores, rel_sets, bands, ks=(5, 10)):
    out = {}
    for band in sorted(set(bands)):
        idx = [i for i, b in enumerate(bands) if b == band]
        sub = ranking_metrics(scores[idx], [rel_sets[i] for i in idx], ks)
        out[band] = {"learners": len(idx), **sub}
    return out


def strategy_ablation(strategies, rel_sets, ks=(5, 10)):
    rows = []
    for name, sc in strategies.items():
        m = ranking_metrics(sc, rel_sets, ks)
        rows.append({"strategy": name, "precision@5": m["precision@5"], "recall@10": m["recall@10"],
                     "ndcg@10": m["ndcg@10"], "map@10": m["map@10"], "mrr": m["mrr"]})
    return sorted(rows, key=lambda r: -r["ndcg@10"])
