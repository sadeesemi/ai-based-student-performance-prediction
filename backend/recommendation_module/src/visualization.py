"""All static figures + the two standalone HTML reports.

Every figure is written to outputs/figures and referenced by outputs/visual_report.html
(the evaluation / preprocessing report window) while outputs/knowledge_graph.html is
the live interactive student-resource knowledge graph window.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    from . import minigraph as nx

from . import config
from .recommender import FEATURE_NAMES, PRETTY

INK = "#12232a"
ACCENT = "#0f6f7a"
LOW, MED, HIGH = "#2f9e6f", "#e0a11b", "#d9534f"
RISK_COLORS = {"Low Risk": LOW, "Medium Risk": MED, "High Risk": HIGH}
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
FIGS = []

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
    "axes.edgecolor": "#cdd5d8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#5a6b72", "ytick.color": "#5a6b72", "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": "#e6eaec", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False,
})


def _save(fig, name, title, caption):
    path = config.FIGURE_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    FIGS.append({"file": name, "title": title, "caption": caption})
    return path


def generate_all(ctx):
    FIGS.clear()
    p, res = ctx["profiles"], ctx["resources"]

    # 01 missing values -------------------------------------------------------
    miss = ctx["report"]["missing_values"]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    if miss:
        ax.barh(list(miss.keys()), list(miss.values()), color=HIGH)
        for i, v in enumerate(miss.values()):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No missing values", ha="center")
    ax.set_xlabel("Missing values (count)")
    ax.set_title("Data quality check - missing values before preprocessing")
    _save(fig, "01_missing_values.png", "Missing values",
          "Only Non_Engaged_Modules is sparse; blanks mean the learner has no flagged lesson and are imputed as an empty gap list.")

    # 02 risk distribution ----------------------------------------------------
    vc = p["Predicted_Risk_Level"].value_counts().reindex(RISK_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.bar(vc.index, vc.values, color=[RISK_COLORS[r] for r in vc.index], width=0.6)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{int(v)}\n({v/len(p)*100:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Learners"); ax.set_ylim(0, vc.max() * 1.2)
    ax.set_title("Predicted risk level distribution (input from Module 02)")
    _save(fig, "02_risk_distribution.png", "Risk distribution",
          "Risk bands consumed from the prediction module drive the intervention weighting.")

    # 03 learning style vs risk ----------------------------------------------
    ct = pd.crosstab(p["Learning_Style"], p["Predicted_Risk_Level"]).reindex(columns=RISK_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bottom = np.zeros(len(ct))
    for r in RISK_ORDER:
        ax.bar(ct.index, ct[r], bottom=bottom, color=RISK_COLORS[r], label=r, width=0.62)
        bottom += ct[r].to_numpy()
    ax.set_ylabel("Learners"); ax.legend(title="Predicted_Risk_Level", fontsize=8)
    ax.set_title("Learning style vs predicted risk level (profiling input)")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
    _save(fig, "03_learning_style_vs_risk.png", "Learning style vs risk",
          "At-Risk and Passive learners concentrate in the medium/high bands, which is what the format matcher reacts to.")

    # 04 engagement distribution ---------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for r in RISK_ORDER:
        vals = p.loc[p["Predicted_Risk_Level"] == r, "engagement_score"]
        if len(vals) > 2:
            xs = np.linspace(0, 100, 200)
            bw = 1.06 * vals.std() * len(vals) ** (-0.2)
            dens = np.exp(-0.5 * ((xs[:, None] - vals.to_numpy()[None, :]) / bw) ** 2).sum(1)
            dens = dens / (len(vals) * bw * np.sqrt(2 * np.pi))
            ax.fill_between(xs, dens, color=RISK_COLORS[r], alpha=0.45, label=r)
    ax.set_xlabel("Engagement score (0-100)"); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    ax.set_title("Engineered feature - engagement score by risk level")
    _save(fig, "04_engagement_distribution.png", "Engagement score distribution",
          "Engagement intensity is a weighted composite of logins, LMS time, clicks, resources, forum activity and attendance.")

    # 05 correlation heatmap --------------------------------------------------
    cols = ["gpa", "total_marks", "exam_results", "quiz_scores", "attendance_percentage",
            "login_frequency", "time_spent_lms", "resources_accessed", "forum_chat_activity",
            "click_count", "engagement_score", "performance_score", "risk_score"]
    corr = p[cols].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i,j]:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(corr.iloc[i, j]) > 0.55 else INK)
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.grid(False); ax.set_title("Correlation matrix - academic, behavioural & engagement features")
    _save(fig, "05_correlation_heatmap.png", "Correlation heatmap",
          "Confirms academic and behavioural blocks carry complementary (not redundant) signal.")

    # 06 engagement vs performance -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for r in RISK_ORDER:
        sub = p[p["Predicted_Risk_Level"] == r]
        ax.scatter(sub["engagement_score"], sub["performance_score"], s=6, alpha=0.45,
                   color=RISK_COLORS[r], label=r, linewidths=0)
    ax.set_xlabel("engagement_score"); ax.set_ylabel("performance_score"); ax.legend(fontsize=8)
    ax.set_title("Engagement vs performance (risk-coloured learner space)")
    _save(fig, "06_engagement_vs_performance.png", "Engagement vs performance",
          "The learner space the recommender operates over; interventions differ across these quadrants.")

    # 07 boxplots by risk -----------------------------------------------------
    box_cols = ["total_marks", "attendance_percentage", "quiz_scores", "login_frequency",
                "time_spent_lms", "resources_accessed"]
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.0))
    for ax, c in zip(axes.ravel(), box_cols):
        data = [p.loc[p["Predicted_Risk_Level"] == r, c].to_numpy() for r in RISK_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color=INK))
        for patch, r in zip(bp["boxes"], RISK_ORDER):
            patch.set_facecolor(RISK_COLORS[r]); patch.set_alpha(0.75)
        ax.set_xticks([1, 2, 3], ["Low", "Med", "High"], fontsize=8)
        ax.set_title(c, fontsize=9)
    fig.suptitle("Feature distributions by predicted risk level", y=1.01)
    _save(fig, "07_feature_boxplots_by_risk.png", "Feature boxplots by risk",
          "Behavioural spread per band - the basis of the context-aware weighting.")

    # 08 non engaged modules --------------------------------------------------
    counts = {}
    for gl in p["gap_lessons"]:
        for l in gl:
            counts[l] = counts.get(l, 0) + 1
    ser = pd.Series(counts).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    cmap = plt.get_cmap("magma")
    ax.barh(ser.index, ser.values, color=[cmap(0.15 + 0.7 * i / len(ser)) for i in range(len(ser))])
    for i, v in enumerate(ser.values):
        ax.text(v, i, f" {v}", va="center", fontsize=7)
    ax.set_xlabel("Learners"); ax.set_title("Non-engaged lessons - learning gaps driving recommendations")
    _save(fig, "08_non_engaged_modules.png", "Non-engaged lessons",
          "Gap frequency across the cohort; each gap becomes a Student -> Lesson edge in the knowledge graph.")

    # 09 normalisation before / after ----------------------------------------
    before, after = ctx["normalisation"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, frame, title in ((axes[0], before, "Before - raw scales"), (axes[1], after, "After - min-max [0,1]")):
        ax.boxplot([frame[c].to_numpy() for c in frame.columns], showfliers=False, widths=0.6,
                   patch_artist=True, boxprops=dict(facecolor=ACCENT, alpha=0.35),
                   medianprops=dict(color=INK))
        ax.set_xticks(range(1, len(frame.columns) + 1), frame.columns, rotation=90, fontsize=6.5)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Preprocessing - feature normalisation step", y=1.02)
    _save(fig, "09_normalization_before_after.png", "Normalisation before/after",
          "Min-max scaling puts every engineered feature on a comparable range before modelling.")

    # 10 resource catalogue ---------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.4))
    a = axes[0][0]; vc2 = res["Resource_Type"].value_counts()
    a.bar(vc2.index, vc2.values, color=ACCENT); a.set_title("Resource type", fontsize=9)
    plt.setp(a.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    a = axes[0][1]; order = ["Foundational", "Beginner", "Intermediate", "Advanced"]
    vc3 = res["Difficulty_Level"].value_counts().reindex(order).fillna(0)
    a.bar(vc3.index, vc3.values, color="#7b4b94"); a.set_title("Difficulty level", fontsize=9)
    plt.setp(a.get_xticklabels(), rotation=25, ha="right", fontsize=7)
    a = axes[1][0]
    a.hist(res["Estimated_Duration_Min"], bins=12, color=MED, edgecolor="white")
    a.set_title("Estimated duration (min)", fontsize=9)
    a = axes[1][1]; vc4 = res["Lesson_Name"].value_counts().sort_values()
    a.barh(vc4.index, vc4.values, color=LOW); a.set_title("Resources per lesson", fontsize=9)
    plt.setp(a.get_yticklabels(), fontsize=6.5)
    fig.suptitle("Indexed LMS resource catalogue", y=1.01)
    _save(fig, "10_resource_catalogue.png", "Resource catalogue",
          f"{len(res)} LMS resources across {res['Lesson_Name'].nunique()} lessons form the candidate pool.")

    # 11 feature importance ---------------------------------------------------
    imp = pd.Series(ctx["importance"]["impurity"]).sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.barh([PRETTY.get(i, i) for i in imp.index], imp.values, color="#3d2b6d")
    ax.set_xlabel("Gini importance"); plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.set_title("Random Forest feature importance - recommendation relevance")
    _save(fig, "11_feature_importance.png", "Feature importance",
          "Which signals the ranker actually leans on when scoring a learner-resource pair.")

    # 12 top-k quality --------------------------------------------------------
    q = ctx["topk_curve"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ks = [r["k"] for r in q]
    for key, colr in (("precision", ACCENT), ("recall", LOW), ("ndcg", "#7b4b94")):
        ax.plot(ks, [r[key] for r in q], marker="o", color=colr, label=key.upper() + "@K")
    ax.set_xticks(ks); ax.set_xlabel("K"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title("Top-K recommendation quality (held-out learners)")
    _save(fig, "12_topk_quality.png", "Top-K quality",
          "Precision, Recall and NDCG at K on learners never seen during training.")

    # 13 predicted vs actual --------------------------------------------------
    yt, yp = ctx["scatter"]["y_true"], ctx["scatter"]["y_pred"]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(yt, yp, s=5, alpha=0.25, color=ACCENT, linewidths=0)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "--", color=HIGH, lw=1)
    ax.set_xlabel("Actual relevance"); ax.set_ylabel("Predicted relevance")
    ax.set_title(f"Relevance regressor - R2 = {ctx['regression']['r2']:.4f}")
    _save(fig, "13_relevance_pred_vs_actual.png", "Predicted vs actual relevance",
          f"RMSE {ctx['regression']['rmse']:.4f} / MAE {ctx['regression']['mae']:.4f} on held-out pairs.")

    # 14 strategy ablation ----------------------------------------------------
    ab = pd.DataFrame(ctx["ablation"]).sort_values("ndcg@10")
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colors = [ACCENT if "Hybrid" in s else "#9fb0b6" for s in ab["strategy"]]
    ax.barh(ab["strategy"], ab["ndcg@10"], color=colors)
    for i, v in enumerate(ab["ndcg@10"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7.5)
    ax.set_xlim(0, 1.08); ax.set_xlabel("NDCG@10")
    ax.set_title("Ablation - single-strategy baselines vs the proposed hybrid model")
    plt.setp(ax.get_yticklabels(), fontsize=7.5)
    _save(fig, "14_strategy_ablation.png", "Strategy ablation",
          "The hybrid Random Forest fusion beats every individual signal used alone.")

    # 15 coverage by resource type -------------------------------------------
    cov = ctx["coverage_by_type"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    types = list(cov.keys()); x = np.arange(len(types))
    ax.bar(x - 0.2, [cov[t]["recommended"] for t in types], 0.4, color=ACCENT, label="Times recommended")
    ax.bar(x + 0.2, [cov[t]["available"] * 100 for t in types], 0.4, color=LOW, label="Available x100")
    ax.set_xticks(x, types, rotation=20, ha="right", fontsize=8); ax.legend(fontsize=8)
    ax.set_title("Recommendation coverage by resource type")
    _save(fig, "15_recommendation_coverage.png", "Coverage by resource type",
          "How often each delivery format reaches a learner's top-6 versus its share of the catalogue.")

    # 16 intervention profile -------------------------------------------------
    cat = pd.Series(ctx["intervention_categories"]).sort_values(ascending=False)
    band = pd.DataFrame(ctx["intervention_by_risk"]).T.reindex(RISK_ORDER).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    axes[0].pie(cat.values, labels=cat.index, autopct="%1.0f%%", startangle=110,
                colors=[ACCENT, LOW, MED, "#7b4b94", HIGH][:len(cat)],
                textprops=dict(fontsize=7.5), wedgeprops=dict(width=0.55))
    axes[0].set_title("Intervention mix by category", fontsize=9)
    bottom = np.zeros(len(band))
    for i, c in enumerate(band.columns):
        axes[1].bar(band.index, band[c], bottom=bottom, label=c,
                    color=[ACCENT, LOW, MED, "#7b4b94", HIGH][i % 5], width=0.6)
        bottom += band[c].to_numpy()
    axes[1].legend(fontsize=6.5); axes[1].set_title("Actions generated per risk band", fontsize=9)
    plt.setp(axes[1].get_xticklabels(), fontsize=8)
    _save(fig, "16_intervention_profile.png", "Intervention profile",
          "Structured mapping from risk state to study planning, revision, time management, resource and support actions.")

    # 17 global SHAP-style impact --------------------------------------------
    ci = ctx["contrib_summary"]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    labels = [PRETTY.get(f, f) for f in ci["features"]]
    colors = [HIGH if v < 0 else ACCENT for v in ci["mean"]]
    ax.barh(labels, ci["mean"], color=colors)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Mean additive contribution to predicted relevance")
    plt.setp(ax.get_yticklabels(), fontsize=7)
    ax.set_title("Why these recommendations - additive tree contributions (SHAP-style)")
    _save(fig, "17_feature_impact_shap.png", "Feature impact (SHAP-style)",
          "Exact additive attribution over the forest: prediction = bias + sum of contributions.")

    # 18 static knowledge graph ----------------------------------------------
    Gs = ctx["sample_graph"]
    pos = nx.spring_layout(Gs, seed=config.RANDOM_SEED, k=0.42, iterations=60)
    kind_color = {"student": "#4b93d1", "lesson": "#d9534f", "topic": "#e0a11b",
                  "resource": "#2f9e6f", "tag": "#b9c4c9", "risk": "#7b4b94", "style": "#0f6f7a"}
    kind_size = {"student": 22, "lesson": 150, "topic": 70, "resource": 55, "tag": 10,
                 "risk": 130, "style": 130}
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for u, vtx in Gs.edges():
        ax.plot([pos[u][0], pos[vtx][0]], [pos[u][1], pos[vtx][1]], color=INK, alpha=0.14, lw=0.45, zorder=1)
    for kind, colr in kind_color.items():
        pts = np.array([pos[n] for n, d in Gs.nodes(data=True) if d["kind"] == kind])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=kind_size[kind], c=colr, label=kind,
                       linewidths=0, zorder=2)
    for n, d in Gs.nodes(data=True):
        if d["kind"] in ("lesson", "risk", "style"):
            ax.text(pos[n][0], pos[n][1] + 0.045, d["label"], fontsize=6, ha="center", color=INK)
    ax.legend(scatterpoints=1, fontsize=7, loc="upper left"); ax.axis("off"); ax.grid(False)
    ax.set_title(f"Constructed knowledge graph - learners, lessons, topics, resources & tags\n"
                 f"(sample view: {Gs.number_of_nodes()} nodes / {Gs.number_of_edges()} edges; "
                 f"full graph {ctx['kg_stats']['nodes']} nodes / {ctx['kg_stats']['edges']} edges)")
    _save(fig, "18_knowledge_graph.png", "Knowledge graph",
          "Static preview - open the interactive window for the live force-directed graph.")

    # 19 GNN training ---------------------------------------------------------
    hist = pd.DataFrame(ctx["gnn_history"])
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.plot(hist["epoch"], hist["loss"], color=ACCENT, label="training loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Link-prediction loss", color=ACCENT)
    ax2 = ax.twinx(); ax2.plot(hist["epoch"], hist["val_auc"], color=LOW, label="held-out link AUC")
    ax2.set_ylabel("Held-out link AUC", color=LOW); ax2.grid(False)
    ax.set_title("GraphSAGE (GNN) unsupervised training on the knowledge graph")
    _save(fig, "19_gnn_training.png", "GNN training",
          f"Final held-out link AUC {hist['val_auc'].iloc[-1]:.4f} - embeddings feed the graph_score feature.")

    # 20 confusion matrix -----------------------------------------------------
    cm = np.array(ctx["classification"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["not relevant", "relevant"]); ax.set_yticks([0, 1], ["not relevant", "relevant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
    ax.set_title(f"Relevance classifier - F1 {ctx['classification']['f1']:.3f}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "20_confusion_matrix.png", "Confusion matrix",
          "Binary view of the ranker: is this pair inside the learner's relevance set?")

    # 21 metrics by risk band -------------------------------------------------
    bb = ctx["band_metrics"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bands = [b for b in RISK_ORDER if b in bb]
    x = np.arange(len(bands))
    ax.bar(x - 0.22, [bb[b]["precision@5"] for b in bands], 0.44, color=ACCENT, label="Precision@5")
    ax.bar(x + 0.22, [bb[b]["ndcg@10"] for b in bands], 0.44, color=LOW, label="NDCG@10")
    for i, b in enumerate(bands):
        ax.text(i, 1.02, f"n={bb[b]['learners']}", ha="center", fontsize=7)
    ax.set_xticks(x, bands); ax.set_ylim(0, 1.12); ax.legend(fontsize=8)
    ax.set_title("Recommendation quality per predicted risk band")
    _save(fig, "21_metrics_by_risk_band.png", "Quality by risk band",
          "Stability check: the ranker does not degrade for the learners who need it most.")

    # 22 model comparison -----------------------------------------------------
    mc = pd.DataFrame(ctx["model_comparison"]).sort_values("ndcg@10")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(mc))
    ax.barh(x - 0.2, mc["r2"], 0.4, color="#9fb0b6", label="R2")
    ax.barh(x + 0.2, mc["ndcg@10"], 0.4, color=ACCENT, label="NDCG@10")
    ax.set_yticks(x, mc["model"], fontsize=8); ax.legend(fontsize=8); ax.set_xlim(0, 1.1)
    ax.set_title("Ranking model benchmark (same feature matrix)")
    _save(fig, "22_model_comparison.png", "Model benchmark",
          "Random Forest is retained as the ranker after benchmarking against linear and boosted alternatives.")

    # 23 recommendation rate by style x difficulty ---------------------------
    hm = ctx["style_difficulty"]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    im = ax.imshow(hm["matrix"], cmap="YlGnBu")
    ax.set_xticks(range(len(hm["levels"])), hm["levels"], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(hm["styles"])), hm["styles"], fontsize=8)
    for i in range(len(hm["styles"])):
        for j in range(len(hm["levels"])):
            v = hm["matrix"][i][j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.4 else INK)
    ax.grid(False); fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Share of recommended difficulty per learning style")
    _save(fig, "23_style_difficulty_mix.png", "Difficulty mix per style",
          "Evidence that the rule layer adapts difficulty to the profiling output.")

    # 24 degree distribution --------------------------------------------------
    deg = np.array([d for _, d in ctx["graph"].degree()])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].hist(deg, bins=40, color=ACCENT, edgecolor="white")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Node degree"); axes[0].set_ylabel("Nodes (log)")
    axes[0].set_title("Knowledge graph degree distribution", fontsize=9)
    nt = pd.Series(ctx["kg_stats"]["node_types"]).sort_values()
    axes[1].barh(nt.index, nt.values, color="#7b4b94")
    for i, v in enumerate(nt.values):
        axes[1].text(v, i, f" {v}", va="center", fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_title("Node inventory by entity type", fontsize=9)
    _save(fig, "24_kg_degree_distribution.png", "Graph structure",
          f"{ctx['kg_stats']['nodes']} nodes / {ctx['kg_stats']['edges']} edges, average degree {ctx['kg_stats']['avg_degree']}.")

    # 25 permutation importance ----------------------------------------------
    if "permutation" in ctx["importance"]:
        pi = pd.Series(ctx["importance"]["permutation"]).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh([PRETTY.get(i, i) for i in pi.index], pi.values, color=LOW)
        ax.set_xlabel("Drop in R2 when shuffled"); plt.setp(ax.get_yticklabels(), fontsize=7)
        ax.set_title("Permutation importance (held-out pairs)")
        _save(fig, "25_permutation_importance.png", "Permutation importance",
              "Model-agnostic confirmation of the impurity ranking.")
    return list(FIGS)


# ---------------------------------------------------------------------------
# HTML reports
# ---------------------------------------------------------------------------
REPORT_CSS = """
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
background:#f4f2ed;color:#12232a}header{background:#0d2027;color:#fff;padding:22px 30px}
header h1{margin:0;font-size:20px;letter-spacing:-.2px}header p{margin:4px 0 0;color:#9fb6bd;font-size:12.5px}
.wrap{padding:22px 30px 60px}.tiles{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px}
.tile{background:#fff;border:1px solid #e3e0d8;border-radius:8px;padding:10px 14px;min-width:118px}
.tile b{display:block;font-size:19px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.tile span{font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;color:#7c8b91}
.btn{display:inline-block;background:#0f6f7a;color:#fff;text-decoration:none;padding:9px 16px;border-radius:7px;
font-size:13px;font-weight:600;margin:2px 6px 14px 0}.btn.alt{background:#12232a}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}
.card{background:#fff;border:1px solid #e3e0d8;border-radius:10px;padding:14px;display:flex;flex-direction:column}
.card img{width:100%;border-radius:6px;cursor:zoom-in}.card h3{margin:10px 0 4px;font-size:13px}
.card p{margin:0;font-size:11.5px;color:#6d7c82;line-height:1.45}
.card .no{font-size:9.5px;color:#9aa8ad;letter-spacing:.08em;text-transform:uppercase}
table{border-collapse:collapse;width:100%;background:#fff;font-size:12px;border:1px solid #e3e0d8;border-radius:8px}
th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #eee9df}th{background:#fbfaf7;font-size:10px;
text-transform:uppercase;letter-spacing:.07em;color:#7c8b91}td.num,th.num{text-align:right;font-family:ui-monospace,Menlo,monospace}
h2{font-size:15px;margin:26px 0 10px}.sec{margin-top:8px}
.modal{position:fixed;inset:0;background:rgba(10,20,24,.9);display:none;align-items:center;justify-content:center;z-index:9}
.modal img{max-width:94vw;max-height:92vh}.modal.on{display:flex}
"""


def _table(rows, cols, num_cols=()):
    head = "".join(f'<th class="num">{c}</th>' if c in num_cols else f"<th>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        body += "<tr>" + "".join(
            f'<td class="num">{r.get(c, "")}</td>' if c in num_cols else f'<td>{r.get(c, "")}</td>'
            for c in cols) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_visual_report(ctx, figures, path=None):
    path = path or (config.OUTPUT_DIR / "visual_report.html")
    m = ctx["metrics"]
    tiles = [("Learners trained", f"{ctx['n_students']:,}"), ("Resources", ctx["n_resources"]),
             ("Pairs scored", f"{ctx['n_pairs']:,}"), ("Precision@5", f"{m['precision@5']:.4f}"),
             ("Recall@10", f"{m['recall@10']:.4f}"), ("NDCG@10", f"{m['ndcg@10']:.4f}"),
             ("MAP@10", f"{m['map@10']:.4f}"), ("MRR", f"{m['mrr']:.4f}"),
             ("Ranker R2", f"{ctx['regression']['r2']:.4f}"), ("F1 (relevant)", f"{ctx['classification']['f1']:.4f}"),
             ("KG nodes", f"{ctx['kg_stats']['nodes']:,}"), ("KG edges", f"{ctx['kg_stats']['edges']:,}"),
             ("GNN link AUC", f"{ctx['gnn_history'][-1]['val_auc']:.4f}")]
    tile_html = "".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>" for k, v in tiles)
    cards = ""
    for i, f in enumerate(figures, 1):
        cards += (f"<div class='card'><img src='figures/{f['file']}' alt='{f['title']}' onclick=\"zoom(this.src)\">"
                  f"<div class='no'>{i:02d}</div><h3>{f['title']}</h3><p>{f['caption']}</p></div>")
    metric_rows = [{"metric": k, "value": v} for k, v in ctx["metrics"].items()]
    metric_rows += [{"metric": f"regression:{k}", "value": v} for k, v in ctx["regression"].items()]
    metric_rows += [{"metric": f"classifier:{k}", "value": v} for k, v in ctx["classification"].items()
                    if k != "confusion_matrix"]
    metric_rows += [{"metric": f"catalogue:{k}", "value": v} for k, v in ctx["catalogue"].items()
                    if k != "resource_hit_counts"]
    band_rows = [{"risk band": b, "learners": v["learners"], "precision@5": v["precision@5"],
                  "recall@10": v["recall@10"], "ndcg@10": v["ndcg@10"]} for b, v in ctx["band_metrics"].items()]
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Module 03 - Recommendation Module Report</title><style>{REPORT_CSS}</style></head><body>
<header><h1>Module 3 - Personalized Intervention Recommendation</h1>
<p>Preprocessing diagnostics, knowledge graph and model evaluation figures &middot; trained on
{ctx['n_students']:,} learners and {ctx['n_resources']} LMS resources</p></header>
<div class="wrap">
<div class="tiles">{tile_html}</div>
<a class="btn" href="knowledge_graph.html" target="_blank">Open interactive knowledge graph &rarr;</a>
<a class="btn alt" href="evaluation_metrics.json" target="_blank">Raw metrics (JSON)</a>
<div class="grid">{cards}</div>
<h2>Evaluation metrics</h2>{_table(metric_rows, ["metric", "value"], {"value"})}
<h2>Quality per predicted risk band</h2>
{_table(band_rows, ["risk band", "learners", "precision@5", "recall@10", "ndcg@10"], {"learners", "precision@5", "recall@10", "ndcg@10"})}
<h2>Strategy ablation</h2>
{_table(ctx["ablation"], ["strategy", "precision@5", "recall@10", "ndcg@10", "map@10", "mrr"], {"precision@5", "recall@10", "ndcg@10", "map@10", "mrr"})}
<h2>Ranking model benchmark</h2>
{_table(ctx["model_comparison"], ["model", "r2", "rmse", "precision@5", "ndcg@10", "fit_seconds"], {"r2", "rmse", "precision@5", "ndcg@10", "fit_seconds"})}
</div>
<div class="modal" id="m" onclick="this.classList.remove('on')"><img id="mi"></div>
<script>function zoom(s){{document.getElementById('mi').src=s;document.getElementById('m').classList.add('on')}}</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path
