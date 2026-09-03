"""Phase 1.2 / 1.3 - Data preprocessing, quality checks and need analysis.

Produces the learner profile frame consumed by every downstream stage plus a
JSON quality report used by the preprocessing diagnostics figures.
"""
import json
import numpy as np
import pandas as pd

from . import config

NUMERIC_FEATURES = [
    "gpa", "ca_marks", "final_exam_marks", "total_marks", "attendance_percentage",
    "assignment_marks", "exam_results", "quiz_scores", "login_frequency",
    "time_spent_lms", "resources_accessed", "forum_chat_activity", "click_count",
    "assessment_pct", "session_ratio",
]


def parse_module_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() in ("", "nan", "None"):
        return []
    return [p.strip() for p in str(value).split(",") if p.strip()]


def quality_report(raw):
    missing = raw.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    return {
        "rows": int(len(raw)),
        "columns": int(raw.shape[1]),
        "duplicate_student_ids": int(raw["student_id"].duplicated().sum()),
        "missing_values": {k: int(v) for k, v in missing.items()},
        "missing_total": int(missing.sum()),
        "numeric_columns": int(raw.select_dtypes(include=[np.number]).shape[1]),
        "categorical_columns": int(raw.select_dtypes(exclude=[np.number]).shape[1]),
    }


def _minmax(series):
    lo, hi = float(series.min()), float(series.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def build_profiles(students):
    """Feature engineering: behavioural, academic and engagement indicators."""
    df = students.copy()

    df["assessment_pct"] = (df["assessment_score"] / df["assessment_max_score"].replace(0, np.nan) * 100).fillna(0)
    df["session_ratio"] = (df["sessions_attended"] / df["total_sessions_held"].replace(0, np.nan)).fillna(0)
    df["ca_pct"] = df["ca_marks"] / df["ca_weight"].replace(0, np.nan) * 100
    df["final_pct"] = df["final_exam_marks"] / df["final_exam_weight"].replace(0, np.nan) * 100

    due = pd.to_datetime(df["assignment_due_date"], dayfirst=True, errors="coerce")
    sub = pd.to_datetime(df["assignment_submission_date"], dayfirst=True, errors="coerce")
    df["submission_delay_days"] = (sub - due).dt.days.fillna(0)
    df["is_late"] = (df["submission_status"].str.lower().eq("late") | (df["submission_delay_days"] > 0)).astype(int)

    df["gap_lessons"] = df["Non_Engaged_Modules"].map(parse_module_list)
    df["n_gap_lessons"] = df["gap_lessons"].map(len)

    # ---- engagement intensity (behavioural composite, 0-100) ----------------
    eng_parts = {
        "login_frequency": 0.20,
        "time_spent_lms": 0.22,
        "resources_accessed": 0.20,
        "click_count": 0.14,
        "forum_chat_activity": 0.10,
        "attendance_percentage": 0.14,
    }
    engagement = np.zeros(len(df))
    for col, w in eng_parts.items():
        engagement += w * _minmax(df[col]).to_numpy()
    df["engagement_score"] = np.round(engagement * 100, 1)

    # ---- academic performance composite (0-100) ----------------------------
    perf_parts = {"total_marks": 0.40, "exam_results": 0.22, "quiz_scores": 0.16,
                  "ca_pct": 0.12, "assessment_pct": 0.10}
    perf = np.zeros(len(df))
    for col, w in perf_parts.items():
        perf += w * _minmax(df[col]).to_numpy()
    df["performance_score"] = np.round(perf * 100, 1)

    df["interaction_density"] = df["click_count"] / df["time_spent_lms"].clip(lower=1)
    df["resource_appetite"] = _minmax(df["resources_accessed"])
    df["time_budget"] = _minmax(df["time_spent_lms"])
    df["gpa_norm"] = _minmax(df["gpa"])
    df["prior_score"] = df["previous_results"].map(config.PRIOR_SCALE).fillna(0.5)
    df["grade_points"] = df["final_grade"].map(config.GRADE_SCALE).fillna(0.0)
    df["risk_score"] = df["Predicted_Risk_Level"].map(config.RISK_SCORE).fillna(0.55)
    df["risk_weight"] = df["Predicted_Risk_Level"].map(config.RISK_WEIGHT).fillna(0.7)
    df["attendance_ratio"] = df["attendance_percentage"] / 100.0
    df["quiz_gap"] = 1 - _minmax(df["quiz_scores"])
    df["assessment_gap"] = 1 - _minmax(df["assessment_pct"])
    df["exam_gap"] = 1 - _minmax(df["exam_results"])
    df["forum_activity"] = _minmax(df["forum_chat_activity"])
    df["submission_delay"] = _minmax(df["submission_delay_days"].clip(lower=0))

    # learner ability used by the difficulty matcher
    df["ability"] = (0.45 * _minmax(df["performance_score"]) + 0.30 * df["gpa_norm"]
                     + 0.15 * df["prior_score"] + 0.10 * _minmax(df["engagement_score"])).clip(0, 1)

    # intervention priority (Phase 1.3)
    df["intervention_priority"] = (0.45 * df["risk_score"] + 0.25 * (1 - _minmax(df["performance_score"]))
                                   + 0.20 * (1 - _minmax(df["engagement_score"]))
                                   + 0.10 * _minmax(df["n_gap_lessons"])).round(3)
    return df


def normalise_matrix(df, columns=None):
    """Min-max normalisation snapshot (used for the before/after figure)."""
    columns = columns or NUMERIC_FEATURES
    before = df[columns].astype(float)
    after = before.apply(_minmax)
    return before, after


def run(students):
    report = quality_report(students)
    profiles = build_profiles(students)
    report["engaged_learners"] = int((profiles["n_gap_lessons"] == 0).sum())
    report["learners_with_gaps"] = int((profiles["n_gap_lessons"] > 0).sum())
    report["risk_distribution"] = profiles["Predicted_Risk_Level"].value_counts().to_dict()
    report["learning_style_distribution"] = profiles["Learning_Style"].value_counts().to_dict()
    (config.OUTPUT_DIR / "preprocessing_report.json").write_text(json.dumps(report, indent=2))
    return profiles, report
