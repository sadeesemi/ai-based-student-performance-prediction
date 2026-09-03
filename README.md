# Module 03 - Personalized Intervention Recommendation

Part of the **AI-Based Student Performance Prediction System**. This delivery is the
complete, runnable recommendation module: a Python training pipeline that learns from the
full dataset, and a Create-React-App dashboard that shows the trained output for any
learner you search for.

* **Knowledge graph + GNN** (GraphSAGE, mean aggregator, unsupervised link prediction)
* **Random Forest relevance ranker** over a fused rule / content / graph / context feature space
* **NLP matching** with TF-IDF vectorisation and cosine semantic similarity
* **Explainable AI** - additive tree contributions (SHAP-style), permutation importance, LIME surrogate
* **Structured interventions** mapped from risk band + behaviour gaps
* **Two standalone windows** - a live interactive knowledge graph and a full evaluation / preprocessing report

> Module 01 (profiling) and Module 02 (risk prediction) are **not** re-implemented here.
> Their outputs already exist in the dataset as the `Learning_Style` and
> `Predicted_Risk_Level` columns, and this module consumes them as inputs.

---

## 1. Quick start

### Backend - train on the full dataset

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cd recommendation_module
python main.py
```

Runs in roughly **1-2 minutes** on a laptop and writes every artefact into
`backend/recommendation_module/outputs/`, then publishes the dashboard data and the two
HTML windows into `frontend/public/`.

### Frontend - the dashboard

```bash
cd frontend
npm install
npm start          # or just double-click run-dev.bat on Windows
```

Opens <http://localhost:3000>. Login accepts **any** email/password (no role-based access
control by design) and lands on Module 03. Search any student ID (e.g. `ST1008`,
`ST1024`, `ST1047`) or a name to see that learner's trained recommendations.

### CLI check (proves the numbers come from the model)

```bash
cd backend/recommendation_module
python recommend_student.py ST1008 --verify
```

### Optional Flask API

```bash
cd backend
python app.py                     # http://localhost:5000/api/recommendation/health
```

Then create `frontend/.env` with:

```
REACT_APP_USE_BACKEND=true
REACT_APP_API_BASE=http://localhost:5000/api/recommendation
```

Only `src/services/api.js` changes behaviour - no component edits.

---

## 2. Folder structure

```
MODULE3_DELIVERY/
├── backend/
│   ├── app.py                       Flask entry point, registers module blueprints
│   ├── config.py                    shared backend config
│   ├── requirements.txt
│   ├── .env.example
│   ├── shared/                      db + auth helpers shared by all three modules
│   └── recommendation_module/
│       ├── main.py                  ← full training pipeline (run this)
│       ├── recommend_student.py     CLI demo for a single learner
│       ├── routes.py                /api/recommendation/* blueprint
│       ├── service.py               application-layer service used by routes + CLI
│       ├── data/
│       │   ├── student_dataset.csv  5,000 learners (carries Module 01 + 02 output)
│       │   └── resources.csv        41 indexed LMS resources
│       ├── src/
│       │   ├── config.py            all paths, weights and hyper-parameters
│       │   ├── data_loader.py       ingestion + encoding cleanup
│       │   ├── preprocessing.py     quality report, feature engineering, normalisation
│       │   ├── knowledge_graph.py   heterogeneous KG construction + centrality/meta-paths
│       │   ├── minigraph.py         zero-dependency graph engine (networkx fallback)
│       │   ├── gnn_embeddings.py    GraphSAGE in NumPy (forward + backprop + Adam)
│       │   ├── nlp_matching.py      TF-IDF, learner need documents, cosine similarity
│       │   ├── recommender.py       feature fusion, relevance oracle, RF ranker
│       │   ├── intervention_mapper.py  risk state → structured actions
│       │   ├── explainability.py    tree contributions, permutation importance, LIME
│       │   ├── evaluation.py        Precision@K, Recall@K, NDCG@K, MAP@K, MRR, ablation
│       │   ├── visualization.py     25 figures + the evaluation report window
│       │   └── kg_view.py           the live interactive knowledge graph window
│       └── outputs/                 everything the pipeline produces (see §4)
├── frontend/
│   ├── public/
│   │   ├── data/                    meta.json, students_index.json, students/shard_XX.json
│   │   └── reports/                 knowledge_graph.html, visual_report.html, figures/
│   ├── src/
│   │   ├── services/api.js          the single data-access layer
│   │   ├── StudentContext.js        search + selected learner state
│   │   ├── components/              Bits, FeatureImpact, RiskBadge, Sidebar, StudentSearch, TopBar, Icons
│   │   ├── pages/                   Login, Module1Profiling, Module2Prediction, Module3Recommendations
│   │   ├── App.js / App.css / index.css / theme.js
│   └── run-dev.bat
└── .vscode/                         launch + task configs for VS Code
```

---

## 3. How the module works

### Phase 1 - preparation
1. **Ingestion** - both CSVs are loaded, encodings normalised, duplicates dropped.
2. **Quality & preprocessing** - missing-value report, min-max normalisation,
   behavioural feature engineering (engagement index, performance index, interaction
   density, submission delay, attendance ratio, ability, intervention priority).
3. **Need analysis** - the non-engaged lessons, risk band and learning style are turned
   into a learner *need document*.
4. **Knowledge graph** - `Student → Lesson → Topic → Resource → Tag` plus
   `Risk level` and `Learning style` entities, and `REQUIRES` prerequisite edges.
   *No week entity is created* - the learner dataset has no week-level signal, so week
   nodes would not be grounded in data.
5. **Resource mapping (GNN)** - a 2-layer GraphSAGE encoder is trained on the graph with
   an unsupervised link-prediction objective and type-aware negative sampling. The
   resulting 32-d embeddings give the `graph_score` and `gnn_style_affinity` features.

### Phase 2 - recommendation engine
Every learner-resource pair gets a **35-feature** vector across four families:

| Family | Examples |
| --- | --- |
| Rule-based filtering | lesson gap, difficulty fit, duration fit, format fit, prerequisite readiness |
| Content-based (TF-IDF) | need-document cosine similarity, tag overlap, lesson semantic similarity |
| Knowledge graph + GNN | embedding similarity, style affinity, resource centrality, meta-path proximity |
| Context-aware behaviour | risk score, engagement, performance, attendance, GPA, time budget, quiz/assessment gaps |

A **Random Forest Regressor** learns the relevance function on 80% of learners and ranks
the catalogue for all of them. A parallel Random Forest classifier answers the binary
"is this pair relevant?" question for precision/recall/F1 reporting.

**Ground truth.** The dataset ships no click-through log, so the relevance target is a
documented pedagogical oracle (risk-weighted learning gap, difficulty and format
suitability, prerequisite readiness, semantic and graph match) with explicit non-linear
interaction effects plus learner-response noise. The relevance set for evaluation is each
learner's top-6 pairs. Splits are **learner-level**, so no student appears in both train
and hold-out.

### Phase 3 - delivery
Structured interventions are generated per learner across five categories: study
planning, revision support, time management, learning resources and academic support
alerts - each with a priority, status and a data-grounded justification.

---

## 4. What the pipeline produces

`backend/recommendation_module/outputs/`

| File | Contents |
| --- | --- |
| `knowledge_graph.html` | **live interactive graph window** - force-directed, drag/zoom, entity filters, ego-network for any of the 5,000 learners |
| `visual_report.html` | **report window** - all 25 figures, metric tiles, evaluation tables |
| `figures/01..25_*.png` | preprocessing diagnostics, KG structure, GNN training, evaluation charts |
| `recommendations.csv` | every learner's top-6 with relevance, signal and reason |
| `interventions.csv` | every generated action with priority, category and status |
| `student_profiles.csv` | engineered learner profile table |
| `evaluation_metrics.json` | every metric, ablation row and benchmark in one file |
| `feature_importance.csv` | impurity + permutation importance + mean contribution per feature |
| `global_explainability.csv` | per-learner additive contributions (SHAP-style) |
| `lime_local_explanations.csv` | LIME surrogate coefficients for a sampled set of learners |
| `heldout_pair_scores.csv` | predicted vs actual relevance for every held-out pair |
| `strategy_comparison.csv` / `model_comparison.csv` | ablation and model benchmark tables |
| `recommender_rf.joblib`, `relevance_classifier.joblib`, `tfidf_vectorizer.joblib`, `gnn_embeddings.joblib`, `knowledge_graph.joblib` | trained artefacts |
| `dashboard_data.json`, `students_payload.json`, `students_index.json`, `top1_features.csv` | what the dashboard and the API serve |

---

## 5. Results on this dataset

Trained on **5,000 learners x 41 resources = 205,000 pairs** (4,000 learners for training, 1,000 held out).

| Metric | @1 | @3 | @5 | @10 |
| --- | --- | --- | --- | --- |
| PRECISION@K | 0.9910 | 0.9033 | 0.8050 | 0.5111 |
| RECALL@K | 0.1856 | 0.4918 | 0.7204 | 0.9077 |
| NDCG@K | 0.9910 | 0.9294 | 0.8745 | 0.9063 |
| MAP@K | 0.9910 | 0.9878 | 0.9635 | 0.9137 |

* MRR **0.9948** &nbsp;|&nbsp; Hit-rate@10 **1.0000**
* Relevance regressor: **R2 0.8933**, RMSE 0.0408, MAE 0.0325, Spearman 0.8697
* Relevance classifier: accuracy 0.9435, precision 0.8965, recall 0.6663, F1 **0.7644**, ROC-AUC 0.9615
* Knowledge graph: **5,158 nodes / 16,769 edges**, avg degree 6.5, 7 entity types, 8 relation types
* GNN (GraphSAGE, 32-d, 250 epochs): held-out link AUC **0.8237**
* Catalogue coverage 0.9756, intra-list lesson diversity 0.4848

**Strategy ablation (NDCG@10, held-out learners)**

| Strategy | P@5 | R@10 | NDCG@10 | MAP@10 |
| --- | --- | --- | --- | --- |
| Hybrid - Random Forest (proposed) | 0.8050 | 0.9077 | 0.9063 | 0.9137 |
| Rule-based filtering only | 0.7402 | 0.8375 | 0.8489 | 0.8922 |
| Hybrid - fixed linear weights | 0.6862 | 0.8410 | 0.8342 | 0.8525 |
| Content-based only (TF-IDF) | 0.5472 | 0.7418 | 0.7260 | 0.7776 |
| Knowledge graph + GNN only | 0.3430 | 0.4872 | 0.4421 | 0.5148 |
| Popularity / centrality baseline | 0.1654 | 0.2302 | 0.1977 | 0.2953 |

**Signal mix (share of ranker importance)**: Rule-based filtering 77.3%, Knowledge graph + GNN 12.1%, Content-based (TF-IDF) 8.4%, Context-aware behaviour 2.1%

Full tables (per risk band, ablation, model benchmark) are in `visual_report.html` and
`evaluation_metrics.json`.

---

## 6. Notes

* `networkx` is optional. If it is not installed, `src/minigraph.py` provides the same API.
* No deep-learning runtime is needed - the GNN is implemented in NumPy (forward pass,
  manual backprop, Adam) and trains in seconds.
* The dashboard is Create React App (not Vite) with `react-router-dom` page routing, as required.
* Re-running `python main.py` regenerates everything deterministically (seed 42).
