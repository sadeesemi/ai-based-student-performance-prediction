"""Central configuration for the Personalized Intervention Recommendation Module (Module 03)."""
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
MODULE_DIR = SRC_DIR.parent
BACKEND_DIR = MODULE_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent

DATA_DIR = MODULE_DIR / "data"
OUTPUT_DIR = MODULE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

STUDENT_FILE = DATA_DIR / "student_dataset.csv"
RESOURCE_FILE = DATA_DIR / "resources.csv"

# React app (frontend) locations the training pipeline publishes to
FRONTEND_DIR = PROJECT_DIR / "frontend"
FRONTEND_DATA_DIR = FRONTEND_DIR / "public" / "data"
FRONTEND_REPORT_DIR = FRONTEND_DIR / "public" / "reports"

RANDOM_SEED = 42

# ---- recommendation settings -------------------------------------------------
TOP_K = 6                    # resources surfaced per learner in the dashboard
K_VALUES = [1, 3, 5, 10]     # K used for Precision@K / Recall@K / NDCG@K / MAP@K
RELEVANT_PER_STUDENT = 6     # size of the held-out relevance set per learner
RELEVANCE_FLOOR = 0.50       # a pair must clear this to count as relevant
TEST_SIZE = 0.2              # learner-level hold-out (no student leaks across split)

# ---- knowledge graph / GNN ---------------------------------------------------
GNN_DIM = 32
GNN_EPOCHS = 250
GNN_LR = 0.015
KG_SAMPLE_STUDENTS = 80      # learners drawn into the interactive sample network

# ---- random forest ranker ----------------------------------------------------
RF_PARAMS = dict(n_estimators=110, max_depth=13, min_samples_leaf=20,
                 max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED)
RF_CLF_PARAMS = dict(n_estimators=90, max_depth=13, min_samples_leaf=20,
                     max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED)

# ---- domain vocabularies -----------------------------------------------------
DIFFICULTY_SCALE = {"Foundational": 0.15, "Beginner": 0.35, "Intermediate": 0.60, "Advanced": 0.85}
RISK_WEIGHT = {"Low Risk": 0.40, "Medium Risk": 0.70, "High Risk": 1.00}
RISK_SCORE = {"Low Risk": 0.20, "Medium Risk": 0.55, "High Risk": 0.85}
PRIOR_SCALE = {"Weak": 0.25, "Average": 0.50, "Good": 0.75, "Excellent": 1.00}
GRADE_SCALE = {"A+": 4.2, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
               "C+": 2.3, "C": 2.0, "C-": 1.7, "D": 1.0, "F": 0.0}

# learning style -> preferred resource delivery format (rule-based filter layer)
STYLE_TYPE_FIT = {
    "Highly Engaged":   {"URL/Lab": 1.00, "PDF": 0.85, "URL": 0.80, "URL/Video": 0.70, "Video": 0.60},
    "Moderate Learner": {"PDF": 1.00, "URL/Video": 0.90, "URL/Lab": 0.80, "URL": 0.75, "Video": 0.70},
    "Passive Learner":  {"Video": 1.00, "URL/Video": 0.95, "URL": 0.70, "PDF": 0.55, "URL/Lab": 0.50},
    "At-Risk Learner":  {"Video": 1.00, "URL/Video": 0.90, "PDF": 0.70, "URL/Lab": 0.65, "URL": 0.60},
}
STYLE_KEYWORDS = {
    "Highly Engaged": "advanced practical lab hands on project automation extension challenge",
    "Moderate Learner": "structured lecture notes worked example template guided practice",
    "Passive Learner": "short video recording walkthrough summary revision overview",
    "At-Risk Learner": "foundational basics recap remedial recording step by step introduction",
}
RISK_KEYWORDS = {
    "Low Risk": "enrichment advanced extension mastery",
    "Medium Risk": "reinforcement practice consolidation revision",
    "High Risk": "foundational remedial catch up basics urgent support recording",
}

# hybrid strategy families used for the strategy-mix chart
STRATEGY_FAMILIES = {
    "Rule-based filtering": ["rule_score", "lesson_gap", "difficulty_fit", "duration_fit", "style_type_fit", "prereq_readiness"],
    "Content-based (TF-IDF)": ["content_score", "tfidf_lesson_sim", "tag_overlap"],
    "Knowledge graph + GNN": ["graph_score", "kg_proximity", "kg_resource_centrality", "gnn_style_affinity"],
    "Context-aware behaviour": ["risk_score", "engagement_score", "performance_score", "attendance_ratio",
                                "gpa_norm", "prior_score", "time_budget", "interaction_density",
                                "submission_delay", "forum_activity", "n_gap_lessons", "quiz_gap",
                                "assessment_gap", "resource_appetite"],
}

for _d in (OUTPUT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
