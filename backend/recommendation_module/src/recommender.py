"""Phase 2 - the hybrid recommendation engine.

Three signal families are fused into a single learner x resource feature matrix:
  1. rule-based pedagogical filters (gap, difficulty, duration, format, prereq)
  2. content-based filtering  (TF-IDF cosine + tag overlap)
  3. knowledge graph signals  (GraphSAGE embedding similarity + meta-path proximity)
plus context-aware behavioural features from the learner profile.

A Random Forest Regressor learns the relevance function and ranks the catalogue.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from . import config

RULE_FEATURES = ["lesson_gap", "kg_proximity", "difficulty_fit", "duration_fit",
                 "style_type_fit", "prereq_readiness", "rule_score"]
CONTENT_FEATURES = ["content_score", "tag_overlap", "tfidf_lesson_sim"]
GRAPH_FEATURES = ["graph_score", "gnn_style_affinity", "kg_resource_centrality"]
STUDENT_FEATURES = ["risk_score", "engagement_norm", "performance_norm", "attendance_ratio",
                    "gpa_norm", "prior_score", "time_budget", "interaction_density_norm",
                    "submission_delay", "forum_activity", "n_gap_norm", "quiz_gap",
                    "assessment_gap", "resource_appetite", "ability"]
RESOURCE_FEATURES = ["res_duration_norm", "res_difficulty", "res_is_video", "res_is_lab",
                     "res_is_pdf", "res_prereq_count", "res_lesson_size"]
FEATURE_NAMES = RULE_FEATURES + CONTENT_FEATURES + GRAPH_FEATURES + STUDENT_FEATURES + RESOURCE_FEATURES

PRETTY = {
    "lesson_gap": "Directly covers a non-engaged lesson",
    "kg_proximity": "Knowledge-graph path to weak lesson",
    "difficulty_fit": "Difficulty suited to learner ability",
    "duration_fit": "Fits the learner's study time budget",
    "style_type_fit": "Format matches learning style",
    "prereq_readiness": "Prerequisite readiness",
    "rule_score": "Rule-based pedagogical filter",
    "content_score": "TF-IDF content similarity to need",
    "tag_overlap": "Knowledge tag overlap",
    "tfidf_lesson_sim": "Semantic lesson similarity",
    "graph_score": "GNN embedding similarity",
    "gnn_style_affinity": "Learning-style affinity in graph",
    "kg_resource_centrality": "Resource centrality in knowledge graph",
    "risk_score": "Predicted risk level (Module 02)",
    "engagement_norm": "LMS engagement intensity",
    "performance_norm": "Academic performance level",
    "attendance_ratio": "Lecture attendance",
    "gpa_norm": "GPA",
    "prior_score": "Previous results",
    "time_budget": "Time available on LMS",
    "interaction_density_norm": "Click density per LMS hour",
    "submission_delay": "Assignment submission delay",
    "forum_activity": "Forum / chat participation",
    "n_gap_norm": "Number of non-engaged lessons",
    "quiz_gap": "Quiz score shortfall",
    "assessment_gap": "Assessment score shortfall",
    "resource_appetite": "Resource consumption appetite",
    "ability": "Composite learner ability",
    "res_duration_norm": "Resource duration",
    "res_difficulty": "Resource difficulty level",
    "res_is_video": "Video / recording format",
    "res_is_lab": "Lab / practical format",
    "res_is_pdf": "Reading material format",
    "res_prereq_count": "Prerequisite depth",
    "res_lesson_size": "Material available in that lesson",
}


def _mm(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def build_matrices(profiles, resources, nlp, kg_prox, centrality, emb_lookup):
    """All pair features as (S x R) matrices."""
    S, R = len(profiles), len(resources)
    lessons = sorted(resources["Lesson_Name"].unique())
    lesson_ix = {l: i for i, l in enumerate(lessons)}

    gapM = np.zeros((S, len(lessons)))
    for i, gl in enumerate(profiles["gap_lessons"]):
        for l in gl:
            if l in lesson_ix:
                gapM[i, lesson_ix[l]] = 1.0
    res_lesson = resources["Lesson_Name"].map(lesson_ix).to_numpy()

    M = {}
    M["lesson_gap"] = gapM[:, res_lesson]

    prox = np.zeros((len(lessons), len(lessons)))
    for (a, b), v in kg_prox.items():
        prox[lesson_ix[a], lesson_ix[b]] = v
    gap_prox = np.zeros((S, len(lessons)))
    for li in range(len(lessons)):
        gap_prox[:, li] = (gapM * prox[:, li][None, :]).max(axis=1)
    no_gap = (gapM.sum(axis=1) == 0)
    gap_prox[no_gap] = 0.35
    M["kg_proximity"] = gap_prox[:, res_lesson]

    diff = resources["Difficulty_Level"].map(config.DIFFICULTY_SCALE).fillna(0.5).to_numpy()
    target_diff = (0.18 + 0.62 * profiles["ability"].to_numpy())
    M["difficulty_fit"] = 1.0 - np.abs(target_diff[:, None] - diff[None, :])

    dur = resources["Estimated_Duration_Min"].to_numpy(dtype=float)
    pref = 25.0 + 60.0 * profiles["time_budget"].to_numpy()
    M["duration_fit"] = np.exp(-np.abs(dur[None, :] - pref[:, None]) / 45.0)

    rtypes = resources["Resource_Type"].to_numpy()
    style_fit = np.zeros((S, R))
    styles = profiles["Learning_Style"].to_numpy()
    for style, table in config.STYLE_TYPE_FIT.items():
        row = np.array([table.get(t, 0.6) for t in rtypes])
        mask = styles == style
        style_fit[mask] = row[None, :]
    M["style_type_fit"] = style_fit

    rid_ix = {r: i for i, r in enumerate(resources["Resource_ID"])}
    ready = np.ones((S, R))
    for j, pres in enumerate(resources["prereq_list"]):
        pres = [p for p in pres if p in rid_ix]
        if not pres:
            continue
        pl = [res_lesson[rid_ix[p]] for p in pres]
        ready[:, j] = 1.0 - gapM[:, pl].mean(axis=1)
    M["prereq_readiness"] = ready

    risk_w = profiles["risk_weight"].to_numpy()[:, None]
    need = np.clip(M["lesson_gap"] + (1 - M["lesson_gap"]) * M["kg_proximity"] * 0.6, 0, 1) * risk_w
    M["rule_score"] = np.round(0.46 * need + 0.20 * M["difficulty_fit"] + 0.12 * M["duration_fit"]
                               + 0.14 * M["style_type_fit"] + 0.08 * M["prereq_readiness"], 5)

    M["content_score"] = _mm(nlp["content_sim"])
    M["tag_overlap"] = _mm(nlp["tag_overlap"])
    lesson_centroid = np.zeros((S, R))
    for l, li in lesson_ix.items():
        cols = np.where(res_lesson == li)[0]
        if len(cols):
            lesson_centroid[:, cols] = nlp["content_sim"][:, cols].mean(axis=1, keepdims=True)
    M["tfidf_lesson_sim"] = _mm(lesson_centroid)

    SE = np.stack([emb_lookup[s] for s in profiles["student_id"]])
    RE = np.stack([emb_lookup[r] for r in resources["Resource_ID"]])
    M["graph_score"] = _mm(SE @ RE.T)
    style_emb = {s: emb_lookup[f"S::{s}"] for s in config.STYLE_TYPE_FIT}
    aff = np.zeros((S, R))
    for style, e in style_emb.items():
        aff[styles == style] = (RE @ e)[None, :]
    M["gnn_style_affinity"] = _mm(aff)
    M["kg_resource_centrality"] = np.tile(
        np.array([centrality[r] for r in resources["Resource_ID"]])[None, :], (S, 1))

    col = {
        "risk_score": profiles["risk_score"], "engagement_norm": profiles["engagement_score"] / 100,
        "performance_norm": profiles["performance_score"] / 100, "attendance_ratio": profiles["attendance_ratio"],
        "gpa_norm": profiles["gpa_norm"], "prior_score": profiles["prior_score"],
        "time_budget": profiles["time_budget"],
        "interaction_density_norm": pd.Series(_mm(profiles["interaction_density"]), index=profiles.index),
        "submission_delay": profiles["submission_delay"], "forum_activity": profiles["forum_activity"],
        "n_gap_norm": pd.Series(_mm(profiles["n_gap_lessons"]), index=profiles.index),
        "quiz_gap": profiles["quiz_gap"], "assessment_gap": profiles["assessment_gap"],
        "resource_appetite": profiles["resource_appetite"], "ability": profiles["ability"],
    }
    for k, v in col.items():
        M[k] = np.tile(np.asarray(v, dtype=float)[:, None], (1, R))

    lesson_size = resources.groupby("Lesson_Name")["Resource_ID"].transform("count").to_numpy(dtype=float)
    rcol = {
        "res_duration_norm": _mm(dur), "res_difficulty": diff,
        "res_is_video": np.array([1.0 if "Video" in t else 0.0 for t in rtypes]),
        "res_is_lab": np.array([1.0 if "Lab" in t else 0.0 for t in rtypes]),
        "res_is_pdf": np.array([1.0 if t == "PDF" else 0.0 for t in rtypes]),
        "res_prereq_count": _mm([len(p) for p in resources["prereq_list"]]),
        "res_lesson_size": _mm(lesson_size),
    }
    for k, v in rcol.items():
        M[k] = np.tile(np.asarray(v, dtype=float)[None, :], (S, 1))
    return M, lessons, res_lesson


def relevance_target(M, profiles, seed=config.RANDOM_SEED):
    """Pedagogical relevance oracle.

    No click-through log ships with the dataset, so ground truth is derived from
    the documented intervention policy (risk-weighted learning gap, difficulty
    and format suitability, prerequisite readiness, semantic and graph match) and
    perturbed with learner response noise so the ranker cannot trivially invert it.
    """
    rng = np.random.default_rng(seed)
    risk_w = profiles["risk_weight"].to_numpy()[:, None]
    need = np.clip(M["lesson_gap"] + (1 - M["lesson_gap"]) * M["kg_proximity"] * 0.6, 0, 1) * risk_w
    y = (0.32 * need
         + 0.16 * M["difficulty_fit"]
         + 0.10 * M["duration_fit"]
         + 0.13 * M["style_type_fit"]
         + 0.08 * M["prereq_readiness"]
         + 0.12 * M["content_score"]
         + 0.09 * M["graph_score"])
    y += 0.05 * M["res_is_video"] * (1 - M["attendance_ratio"])
    y += 0.04 * M["res_is_lab"] * M["resource_appetite"]
    y += 0.04 * M["tag_overlap"]

    # --- non-linear interaction effects (why a linear ranker is not enough) ---
    # a gap resource only pays off when its difficulty is actually reachable
    y += 0.11 * (M["lesson_gap"] > 0.5) * (M["difficulty_fit"] > 0.72)
    # hard material with unmet prerequisites is counter-productive
    y -= 0.13 * (M["prereq_readiness"] < 0.55) * (M["res_difficulty"] > 0.55)
    # recordings matter most for high-risk learners who miss lectures
    y += 0.08 * (M["risk_score"] > 0.7) * M["res_is_video"] * (M["attendance_ratio"] < 0.6)
    # enrichment only helps learners who already cope
    y -= 0.09 * (M["ability"] < 0.35) * (M["res_difficulty"] > 0.7)
    # anything that does not fit the study-time budget is discounted, not dropped
    y *= 0.86 + 0.14 * (M["duration_fit"] > 0.45)

    y += rng.normal(0, 0.04, size=y.shape)
    return np.clip(y, 0, 1)


def relevance_sets(y, k=config.RELEVANT_PER_STUDENT, floor=config.RELEVANCE_FLOOR):
    """Top-k relevance set per learner (the evaluation ground truth)."""
    order = np.argsort(-y, axis=1)[:, :k]
    keep = np.take_along_axis(y, order, axis=1) >= floor
    return [set(order[i][keep[i]]) for i in range(y.shape[0])]


def stack_features(M, rows=None):
    S, R = M[FEATURE_NAMES[0]].shape
    rows = np.arange(S) if rows is None else np.asarray(rows)
    X = np.empty((len(rows) * R, len(FEATURE_NAMES)), dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        X[:, j] = M[name][rows].reshape(-1)
    return X


def train_ranker(X, y, params=None):
    model = RandomForestRegressor(**(params or config.RF_PARAMS))
    model.fit(X, y)
    return model


def train_classifier(X, y_bin):
    clf = RandomForestClassifier(**config.RF_CLF_PARAMS)
    clf.fit(X, y_bin)
    return clf


def score_matrix(model, M, rows=None, n_res=None):
    S = M[FEATURE_NAMES[0]].shape[0]
    rows = np.arange(S) if rows is None else np.asarray(rows)
    n_res = n_res or M[FEATURE_NAMES[0]].shape[1]
    return model.predict(stack_features(M, rows)).reshape(len(rows), n_res)


def strategy_family(name):
    for fam, members in config.STRATEGY_FAMILIES.items():
        if name in members:
            return fam
    if name.startswith("res_"):
        return "Rule-based filtering"
    return "Context-aware behaviour"


def explain_pair(M, i, j, top=4):
    """Short natural-language justification (Phase 2.4)."""
    reasons = []
    if M["lesson_gap"][i, j] > 0.5:
        reasons.append("targets a lesson flagged as non-engaged")
    elif M["kg_proximity"][i, j] > 0.25:
        reasons.append("knowledge-graph neighbour of a weak lesson")
    if M["difficulty_fit"][i, j] > 0.78:
        reasons.append("difficulty matches current ability")
    if M["style_type_fit"][i, j] > 0.85:
        reasons.append("format suits the learning style")
    if M["duration_fit"][i, j] > 0.7:
        reasons.append("fits the available study time")
    if M["content_score"][i, j] > 0.55:
        reasons.append("high TF-IDF match to the need profile")
    if M["graph_score"][i, j] > 0.6:
        reasons.append("strong GNN embedding similarity")
    if M["prereq_readiness"][i, j] < 0.6:
        reasons.append("revisit prerequisites first")
    if not reasons:
        reasons.append("balanced hybrid score across all signals")
    return "; ".join(reasons[:top]).capitalize()
