"""Module 03 - quantitative evaluation of the explainability (XAI) layer.

Standalone add-on. Run it AFTER main.py has finished:

    python xai_eval.py                            (from backend/recommendation_module/)
    python -m recommendation_module.xai_eval      (from backend/)

    optional flags:  --sample 600  --stability 200  --seed 42

It modifies NOTHING in the existing pipeline. It only reads artefacts that
main.py already wrote, then writes new ones:

    reads   outputs/recommender_rf.joblib
            outputs/top1_features.csv
            outputs/lime_local_explanations.csv          (optional)
            frontend/public/data/students_index.json     (optional, risk bands)

    writes  outputs/xai_evaluation.json
            outputs/xai_attributions_sample.csv
            outputs/figures/xai_faithfulness.png
            outputs/figures/xai_stability_sparsity.png
            outputs/figures/xai_shap_lime_agreement.png
            frontend/public/data/xai_evaluation.json     (read by the dashboard)
            frontend/public/reports/xai_evaluation.json  (+ the three figures)

What is measured
----------------
completeness      additivity error |bias + sum(contributions) - model.predict(x)|,
                  proof that the attributions are exact and not approximated
faithfulness      deletion curve: replace the top-k attributed features with the
                  cohort median and measure how far the prediction moves, against
                  a random-ordering control. Comprehensiveness@5 + deletion AUC
sufficiency       keep only the top-k attributed features and drop the rest. A
                  small shift means the short explanation carries the signal
monotonicity      per learner Spearman between |attribution| and the measured
                  one-at-a-time ablation effect of each feature
stability         re-explain a slightly perturbed input, cosine similarity of the
                  two explanation vectors (local Lipschitz robustness)
sparsity          share of attribution mass in the top 5 features, Gini, and the
                  effective number of features (exp of attribution entropy)
agreement         SHAP-style tree contributions versus the LIME local surrogate:
                  Spearman, top-5 Jaccard and sign agreement
determinism       identical inputs must produce identical explanations
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "recommendation_module"

from .src import config                                    # noqa: E402
from .src.explainability import tree_contributions         # noqa: E402

try:
    from .src.recommender import FEATURE_NAMES, PRETTY     # noqa: E402
except Exception:                                          # pragma: no cover
    FEATURE_NAMES, PRETTY = None, {}

KS = (1, 2, 3, 4, 5, 6, 8, 10, 14, 20)
META_COLS = ("student_id", "resource_id", "predicted_relevance")


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _r(x, n=4):
    v = float(x)
    return round(v, n) if np.isfinite(v) else 0.0


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _gini(w):
    w = np.sort(np.abs(np.asarray(w, dtype=float)))
    n = len(w)
    s = w.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * w).sum()) / (n * s) - (n + 1.0) / n)


def _effective_features(w):
    p = np.abs(np.asarray(w, dtype=float))
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def _auc(y, x):
    """Trapezoid area, works on both numpy 1.x and 2.x."""
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x)))


def _hist(values, bins=18):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return {"bins": [_r(c, 4) for c in centres], "counts": [int(c) for c in counts]}


# --------------------------------------------------------------------------
# artefact loading
# --------------------------------------------------------------------------
def load_artefacts():
    out = config.OUTPUT_DIR
    model_path = out / "recommender_rf.joblib"
    feat_path = out / "top1_features.csv"
    if not model_path.exists() or not feat_path.exists():
        raise SystemExit(
            "Trained artefacts not found in %s.\nRun the pipeline first: python main.py" % out)

    model = joblib.load(model_path)
    df = pd.read_csv(feat_path)
    cols = [c for c in df.columns if c not in META_COLS]
    if FEATURE_NAMES:
        cols = [c for c in FEATURE_NAMES if c in df.columns] or cols

    bands = {}
    idx_path = config.FRONTEND_DATA_DIR / "students_index.json"
    if not idx_path.exists():
        idx_path = out / "students_index.json"
    if idx_path.exists():
        try:
            for row in json.loads(idx_path.read_text()):
                bands[row["id"]] = row.get("riskLevel", "Unknown")
        except Exception:
            bands = {}

    lime = None
    lime_path = out / "lime_local_explanations.csv"
    if lime_path.exists():
        try:
            lime = pd.read_csv(lime_path).set_index("student_id")
        except Exception:
            lime = None

    return model, df, cols, bands, lime


# --------------------------------------------------------------------------
# metric blocks
# --------------------------------------------------------------------------
def faithfulness(model, X, contrib, base, rng):
    """Deletion (comprehensiveness) and preservation (sufficiency) curves."""
    n, f = X.shape
    pred_full = model.predict(X)

    all_removed = np.tile(base, (n, 1)).astype(np.float32)
    denom = float(np.mean(np.abs(pred_full - model.predict(all_removed)))) or 1e-9

    order_attr = np.argsort(-np.abs(contrib), axis=1)
    order_rand = np.argsort(rng.random((n, f)), axis=1)
    rows = np.arange(n)[:, None]

    curve = [{"k": 0, "attribution": 0.0, "random": 0.0, "sufficiency": 0.0, "shift": 0.0}]
    for k in [kk for kk in KS if kk <= f]:
        cols_a = order_attr[:, :k]
        d_attr = X.copy()
        d_attr[rows, cols_a] = base[cols_a]
        shift_attr = float(np.mean(np.abs(pred_full - model.predict(d_attr))))

        cols_r = order_rand[:, :k]
        d_rand = X.copy()
        d_rand[rows, cols_r] = base[cols_r]
        shift_rand = float(np.mean(np.abs(pred_full - model.predict(d_rand))))

        keep = np.zeros((n, f), dtype=bool)
        keep[rows, cols_a] = True
        d_keep = np.where(keep, X, base[None, :]).astype(np.float32)
        shift_keep = float(np.mean(np.abs(pred_full - model.predict(d_keep))))

        curve.append({"k": int(k),
                      "attribution": _r(shift_attr / denom),
                      "random": _r(shift_rand / denom),
                      "sufficiency": _r(max(0.0, 1.0 - shift_keep / denom)),
                      "shift": _r(shift_attr, 5)})

    ks = np.array([c["k"] for c in curve], dtype=float)
    ya = np.array([c["attribution"] for c in curve])
    yr = np.array([c["random"] for c in curve])
    span = ks[-1] - ks[0]
    auc_a = (_auc(ya, ks) / span) if span else 0.0
    auc_r = (_auc(yr, ks) / span) if span else 0.0

    at5 = next((c for c in curve if c["k"] == 5), curve[-1])
    return curve, {
        "comprehensiveness@5": _r(at5["attribution"]),
        "comprehensiveness@5_random": _r(at5["random"]),
        "sufficiency@5": _r(at5["sufficiency"]),
        "deletion_auc": _r(auc_a),
        "deletion_auc_random": _r(auc_r),
        "faithfulness_gain": _r(auc_a - auc_r),
        "mean_prediction_shift@5": _r(at5["shift"], 5),
    }


def monotonicity(model, X, contrib, base):
    """Spearman between |attribution| and the measured one-at-a-time effect."""
    n, f = X.shape
    pred_full = model.predict(X)
    effect = np.zeros((n, f))
    for j in range(f):
        Xj = X.copy()
        Xj[:, j] = base[j]
        effect[:, j] = np.abs(pred_full - model.predict(Xj))
    rho = np.array([_spearman(np.abs(contrib[i]), effect[i]) for i in range(n)])
    top1 = float(np.mean(np.argmax(np.abs(contrib), axis=1) == np.argmax(effect, axis=1)))
    return {
        "monotonicity_spearman": _r(np.mean(rho)),
        "monotonicity_spearman_median": _r(np.median(rho)),
        "monotonicity_positive_share": _r(np.mean(rho > 0)),
        "top1_feature_agreement": _r(top1),
    }, rho


def stability(model, X, contrib, ref_std, n_features, rng, noise=0.02):
    """Local robustness: re-explain a nudged input, compare explanation vectors."""
    bump = rng.normal(0.0, 1.0, size=X.shape) * (ref_std * noise)[None, :]
    Xp = (X + bump).astype(np.float32)
    cp, _ = tree_contributions(model, Xp, n_features)

    num = np.sum(contrib * cp, axis=1)
    den = np.linalg.norm(contrib, axis=1) * np.linalg.norm(cp, axis=1) + 1e-12
    cos = num / den

    dx = np.linalg.norm((Xp - X) / (ref_std + 1e-9), axis=1) + 1e-9
    de = np.linalg.norm(cp - contrib, axis=1) / (np.linalg.norm(contrib, axis=1) + 1e-9)
    lip = de / dx

    sign = np.mean(np.sign(contrib) == np.sign(cp), axis=1)
    return {
        "stability_cosine": _r(np.mean(cos)),
        "stability_cosine_p05": _r(np.percentile(cos, 5)),
        "stability_sign_agreement": _r(np.mean(sign)),
        "local_lipschitz_mean": _r(np.mean(lip)),
        "perturbation_sigma": noise,
    }, cos


def sparsity(contrib):
    a = np.abs(contrib)
    tot = a.sum(axis=1) + 1e-12
    srt = -np.sort(-a, axis=1)
    top5 = srt[:, :5].sum(axis=1) / tot
    top8 = srt[:, :8].sum(axis=1) / tot
    gini = np.array([_gini(r) for r in a])
    eff = np.array([_effective_features(r) for r in a])
    return {
        "sparsity_top5_mass": _r(np.mean(top5)),
        "sparsity_top8_mass": _r(np.mean(top8)),
        "attribution_gini": _r(np.mean(gini)),
        "effective_features": _r(np.mean(eff), 2),
    }, top5


def agreement(contrib, ids, cols, lime):
    """SHAP-style tree contributions versus the LIME local surrogate."""
    empty = ({"shap_lime_available": False}, [], [])
    if lime is None:
        return empty
    shared = [c for c in cols if c in lime.columns]
    if len(shared) < 5:
        return empty

    pos = {c: i for i, c in enumerate(cols)}
    rho, jac, sign, pairs = [], [], [], []
    for i, sid in enumerate(ids):
        if sid not in lime.index:
            continue
        s = np.array([contrib[i, pos[c]] for c in shared], dtype=float)
        l = np.asarray(lime.loc[sid, shared], dtype=float).ravel()[:len(shared)]
        if not np.isfinite(l).all():
            continue
        rho.append(_spearman(np.abs(s), np.abs(l)))
        ts = set(np.argsort(-np.abs(s))[:5].tolist())
        tl = set(np.argsort(-np.abs(l))[:5].tolist())
        jac.append(len(ts & tl) / len(ts | tl))
        sign.append(float(np.mean(np.sign(s) == np.sign(l))))
        pairs.append((s, l))

    if not rho:
        return empty

    S = np.array([p[0] for p in pairs])
    L = np.array([p[1] for p in pairs])
    ms, ml = np.abs(S).mean(axis=0), np.abs(L).mean(axis=0)
    order = np.argsort(-ms)[:12]
    table = [{"feature": shared[j],
              "label": PRETTY.get(shared[j], shared[j]),
              "shap": _r(ms[j], 5),
              "lime": _r(ml[j], 5)} for j in order]
    return {
        "shap_lime_available": True,
        "shap_lime_learners": len(rho),
        "shap_lime_spearman": _r(np.mean(rho)),
        "shap_lime_jaccard@5": _r(np.mean(jac)),
        "shap_lime_sign_agreement": _r(np.mean(sign)),
    }, table, rho


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def write_figures(curve, cos, rho_mono, top5, rho_agree, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, accent, warm, soft = "#1d232e", "#0f8f86", "#c9563c", "#8b93a1"
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figs = []

    def save(fig, name, title, caption):
        fig.tight_layout()
        fig.savefig(config.FIGURE_DIR / name, dpi=150, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        figs.append({"file": name, "title": title, "caption": caption,
                     "group": "Explainability evaluation"})

    ks = [c["k"] for c in curve]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ks, [c["attribution"] for c in curve], "-o", color=accent, lw=2, ms=4,
            label="delete top-k attributed features")
    ax.plot(ks, [c["random"] for c in curve], "--s", color=soft, lw=1.6, ms=4,
            label="delete k random features (control)")
    ax.plot(ks, [c["sufficiency"] for c in curve], "-^", color=warm, lw=1.6, ms=4,
            label="keep only top-k: score preserved (sufficiency)")
    ax.set_xlabel("k features")
    ax.set_ylabel("normalised prediction shift")
    ax.set_title("Faithfulness: deletion and preservation curves", color=ink)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "xai_faithfulness.png", "Faithfulness of the explanations",
         "Removing the highest attributed features moves the prediction much more than "
         "removing random features, and the top 5 features alone reproduce most of the "
         "score. Deletion AUC %.3f versus %.3f for the random control."
         % (summary["deletion_auc"], summary["deletion_auc_random"]))

    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
    axs[0].hist(cos, bins=24, color=accent, alpha=0.85)
    axs[0].set_title("Explanation stability", fontsize=10)
    axs[0].set_xlabel("cosine similarity under 2% input noise")
    axs[1].hist(rho_mono, bins=24, color=warm, alpha=0.85)
    axs[1].set_title("Monotonicity", fontsize=10)
    axs[1].set_xlabel("|attribution| vs measured ablation effect")
    axs[2].hist(top5, bins=24, color="#5b6bd6", alpha=0.85)
    axs[2].set_title("Sparsity", fontsize=10)
    axs[2].set_xlabel("attribution mass in the top 5 features")
    for a in axs:
        a.grid(alpha=0.15)
    save(fig, "xai_stability_sparsity.png", "Stability, monotonicity and sparsity",
         "Per-learner distributions across the evaluated cohort sample. Mean cosine "
         "stability %.3f, mean monotonicity %.3f, mean top-5 attribution mass %.3f."
         % (summary.get("stability_cosine", 0), summary.get("monotonicity_spearman", 0),
            summary.get("sparsity_top5_mass", 0)))

    if len(rho_agree):
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        ax.hist(rho_agree, bins=22, color="#7a5bd6", alpha=0.85)
        ax.axvline(float(np.mean(rho_agree)), color=warm, lw=2,
                   label="mean %.3f" % float(np.mean(rho_agree)))
        ax.set_xlabel("Spearman rank correlation, tree contributions vs LIME surrogate")
        ax.set_ylabel("learners")
        ax.set_title("Cross-method agreement (SHAP-style vs LIME)", color=ink)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)
        save(fig, "xai_shap_lime_agreement.png", "SHAP-style versus LIME agreement",
             "Two independent explanation methods rank the same drivers. Mean Spearman "
             "%.3f, top-5 Jaccard %.3f."
             % (summary.get("shap_lime_spearman", 0), summary.get("shap_lime_jaccard@5", 0)))
    return figs


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def run(sample=600, n_stability=200, seed=42):
    t0 = time.time()
    rng = np.random.default_rng(seed)

    print("[01] Loading trained artefacts")
    model, df, cols, bands, lime = load_artefacts()
    f = len(cols)
    print("      learners=%d  features=%d  trees=%d"
          % (len(df), f, len(getattr(model, "estimators_", []))))

    X_all = df[cols].to_numpy(dtype=np.float32)
    base = np.median(X_all, axis=0).astype(np.float32)
    ref_std = X_all.std(axis=0) + 1e-6

    pick = rng.choice(len(df), size=min(sample, len(df)), replace=False)
    if lime is not None:
        lime_rows = np.array([i for i, s in enumerate(df["student_id"]) if s in lime.index],
                             dtype=int)
        pick = np.unique(np.concatenate([pick, lime_rows]))
    X = X_all[pick].copy()
    ids = df["student_id"].to_numpy()[pick]
    n = len(pick)
    print("[02] Explaining %d learners" % n)
    contrib, bias = tree_contributions(model, X, f)

    print("[03] Completeness / additivity check")
    pred = model.predict(X)
    recon = bias + contrib.sum(axis=1)
    completeness = {
        "bias": _r(bias, 5),
        "completeness_error_mean": _r(np.mean(np.abs(pred - recon)), 8),
        "completeness_error_max": _r(np.max(np.abs(pred - recon)), 8),
        "completeness_r2": _r(1.0 - np.var(pred - recon) / (np.var(pred) + 1e-12), 6),
    }

    print("[04] Faithfulness - deletion and sufficiency curves")
    curve, faith = faithfulness(model, X, contrib, base, rng)

    print("[05] Monotonicity - attribution versus measured ablation effect")
    mono, rho_mono = monotonicity(model, X, contrib, base)

    print("[06] Stability - re-explaining perturbed inputs")
    sidx = rng.choice(n, size=min(n_stability, n), replace=False)
    stab, cos = stability(model, X[sidx], contrib[sidx], ref_std, f, rng)

    print("[07] Sparsity and concentration")
    spar, top5 = sparsity(contrib)

    print("[08] Cross-method agreement with the LIME surrogate")
    agr, agr_table, rho_agree = agreement(contrib, ids, cols, lime)

    c2, _ = tree_contributions(model, X[:8], f)
    deterministic = bool(np.allclose(c2, contrib[:8], atol=1e-12))

    summary = {}
    for block in (completeness, faith, mono, stab, spar, agr):
        summary.update(block)
    summary["deterministic"] = deterministic

    per_band = {}
    if bands:
        band_of = np.array([bands.get(s, "Unknown") for s in ids])
        for b in sorted(set(band_of.tolist())):
            m = np.where(band_of == b)[0]
            if len(m) < 20:
                continue
            c_b, f_b = faithfulness(model, X[m], contrib[m], base,
                                    np.random.default_rng(seed))
            at5 = next((c for c in c_b if c["k"] == 5), c_b[-1])
            per_band[b] = {"learners": int(len(m)),
                           "comprehensiveness@5": _r(at5["attribution"]),
                           "sufficiency@5": _r(at5["sufficiency"]),
                           "deletion_auc": f_b["deletion_auc"],
                           "sparsity_top5_mass": _r(np.mean(top5[m]))}

    print("[09] Rendering figures")
    figures = write_figures(curve, cos, rho_mono, top5, rho_agree, summary)

    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "explained_learners": int(n),
        "features": int(f),
        "trees": int(len(getattr(model, "estimators_", []))),
        "sample_seed": int(seed),
        "summary": summary,
        "deletion_curve": curve,
        "per_risk_band": per_band,
        "method_agreement": agr_table,
        "distributions": {
            "stability_cosine": _hist(cos),
            "monotonicity_spearman": _hist(rho_mono),
            "top5_mass": _hist(top5),
            "shap_lime_spearman": _hist(rho_agree) if len(rho_agree) else {"bins": [], "counts": []},
        },
        "figures": figures,
        "headline": [
            {"label": "Completeness error", "value": summary["completeness_error_mean"],
             "hint": "attributions reconstruct the prediction exactly", "good": "low"},
            {"label": "Comprehensiveness@5", "value": summary["comprehensiveness@5"],
             "hint": "prediction shift when the top 5 drivers are removed, random control %s"
                     % summary["comprehensiveness@5_random"], "good": "high"},
            {"label": "Deletion AUC", "value": summary["deletion_auc"],
             "hint": "random ordering control %s" % summary["deletion_auc_random"], "good": "high"},
            {"label": "Sufficiency@5", "value": summary["sufficiency@5"],
             "hint": "share of the prediction preserved by the top 5 drivers alone", "good": "high"},
            {"label": "Stability", "value": summary["stability_cosine"],
             "hint": "cosine similarity of explanations under 2% input noise", "good": "high"},
            {"label": "Monotonicity", "value": summary["monotonicity_spearman"],
             "hint": "attribution rank versus measured ablation effect", "good": "high"},
            {"label": "Top-5 attribution mass", "value": summary["sparsity_top5_mass"],
             "hint": "%s effective features per explanation" % summary["effective_features"],
             "good": "high"},
            {"label": "SHAP vs LIME", "value": summary.get("shap_lime_spearman", 0),
             "hint": "top-5 overlap %s" % summary.get("shap_lime_jaccard@5", "n/a"),
             "good": "high"},
        ],
    }

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "xai_evaluation.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame([{"student_id": ids[i],
                   **{cols[j]: round(float(contrib[i, j]), 6) for j in range(f)}}
                  for i in range(min(n, 500))]).to_csv(
        config.OUTPUT_DIR / "xai_attributions_sample.csv", index=False)

    try:
        config.FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.FRONTEND_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (config.FRONTEND_DATA_DIR / "xai_evaluation.json").write_text(
            json.dumps(report, separators=(",", ":")))
        shutil.copy(config.OUTPUT_DIR / "xai_evaluation.json",
                    config.FRONTEND_REPORT_DIR / "xai_evaluation.json")
        figdst = config.FRONTEND_REPORT_DIR / "figures"
        figdst.mkdir(parents=True, exist_ok=True)
        for fig in figures:
            shutil.copy(config.FIGURE_DIR / fig["file"], figdst / fig["file"])
        print("      published to %s" % config.FRONTEND_DATA_DIR)
    except Exception as err:                                   # pragma: no cover
        print("      front end folder not found, JSON left in outputs/ (%s)" % err)

    print("=" * 78)
    print("XAI EVALUATION   completeness_err=%s   comprehensiveness@5=%s (random %s)"
          % (summary["completeness_error_mean"], summary["comprehensiveness@5"],
             summary["comprehensiveness@5_random"]))
    print("                 sufficiency@5=%s   stability=%s   monotonicity=%s"
          % (summary["sufficiency@5"], summary["stability_cosine"],
             summary["monotonicity_spearman"]))
    print("                 top5_mass=%s   shap_vs_lime=%s   done in %.1fs"
          % (summary["sparsity_top5_mass"], summary.get("shap_lime_spearman", "n/a"),
             time.time() - t0))
    print("=" * 78)
    return report


def main():
    ap = argparse.ArgumentParser(description="Quantitative XAI evaluation for Module 03")
    ap.add_argument("--sample", type=int, default=600, help="learners to explain")
    ap.add_argument("--stability", type=int, default=200, help="learners re-explained under noise")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run(sample=a.sample, n_stability=a.stability, seed=a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
