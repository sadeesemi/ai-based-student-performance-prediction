"""Module 03 - end to end training pipeline.

    python -m recommendation_module.main        (from the backend/ folder)
    python main.py                             (from backend/recommendation_module/)

Trains on the full dataset (every learner in student_dataset.csv x every LMS
resource in resources.csv) and writes every artefact the React front end and the
two report windows need into outputs/ and frontend/public/.
"""
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

from .src import (config, data_loader, evaluation, explainability, gnn_embeddings,
                  intervention_mapper, kg_view, knowledge_graph, nlp_matching,
                  preprocessing, recommender, visualization)
from .src.recommender import FEATURE_NAMES, PRETTY

STEP = 0


def step(msg):
    global STEP
    STEP += 1
    print(f"[{STEP:02d}] {msg}", flush=True)


def main():
    t0 = time.time()
    rng = np.random.default_rng(config.RANDOM_SEED)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    step("Loading datasets (Module 01 + Module 02 outputs are columns of the learner dataset)")
    students, resources = data_loader.load_all()
    print(f"      learners={len(students)}  resources={len(resources)}")

    step("Preprocessing, feature engineering and need analysis")
    profiles, report = preprocessing.run(students)
    profiles = profiles.reset_index(drop=True)
    norm_before, norm_after = preprocessing.normalise_matrix(profiles)

    step("Building the knowledge graph")
    G = knowledge_graph.build(profiles, resources)
    kg_stats = knowledge_graph.stats(G)
    kg_stats["students"] = int(len(profiles))
    centrality = knowledge_graph.resource_centrality(G, resources)
    lesson_prox = knowledge_graph.lesson_proximity(resources)
    print(f"      nodes={kg_stats['nodes']} edges={kg_stats['edges']} avg_degree={kg_stats['avg_degree']}")

    step("Training the GraphSAGE (GNN) resource mapper on the knowledge graph")
    gnn = gnn_embeddings.train(G)
    emb_lookup = {n: e for n, e in zip(gnn["nodes"], gnn["embeddings"])}

    step("NLP layer - TF-IDF vectorisation and semantic need matching")
    nlp = nlp_matching.fit(profiles, resources)

    step("Assembling the learner x resource feature tensor")
    M, lessons, res_lesson = recommender.build_matrices(profiles, resources, nlp, lesson_prox,
                                                        centrality, emb_lookup)
    S, R = len(profiles), len(resources)
    y = recommender.relevance_target(M, profiles)
    rel_sets = recommender.relevance_sets(y)
    print(f"      pairs={S*R:,}  mean relevance={y.mean():.4f}  relevant/learner={np.mean([len(r) for r in rel_sets]):.2f}")

    step("Learner-level train / hold-out split")
    idx = np.arange(S)
    rng.shuffle(idx)
    n_test = int(config.TEST_SIZE * S)
    test_rows, train_rows = np.sort(idx[:n_test]), np.sort(idx[n_test:])
    X_train = recommender.stack_features(M, train_rows)
    y_train = y[train_rows].reshape(-1)
    X_test = recommender.stack_features(M, test_rows)
    y_test = y[test_rows].reshape(-1)
    print(f"      train learners={len(train_rows)} ({len(y_train):,} pairs) | hold-out learners={len(test_rows)}")

    step("Training the Random Forest relevance ranker")
    t = time.time()
    model = recommender.train_ranker(X_train, y_train)
    print(f"      fitted in {time.time()-t:.1f}s")

    step("Training the relevance classifier (relevant vs not relevant)")
    thr_train = np.array([[1.0 if j in rel_sets[i] else 0.0 for j in range(R)] for i in train_rows]).reshape(-1)
    thr_test = np.array([[1.0 if j in rel_sets[i] else 0.0 for j in range(R)] for i in test_rows]).reshape(-1)
    clf = recommender.train_classifier(X_train, thr_train)

    step("Evaluating on held-out learners")
    y_pred_test = model.predict(X_test)
    regression = evaluation.regression_metrics(y_test, y_pred_test)
    scores_test = y_pred_test.reshape(len(test_rows), R)
    rel_test = [rel_sets[i] for i in test_rows]
    metrics = evaluation.ranking_metrics(scores_test, rel_test, config.K_VALUES)
    catalogue = evaluation.catalogue_metrics(scores_test, resources, config.TOP_K)
    proba = clf.predict_proba(X_test)[:, 1]
    classification = evaluation.classification_metrics(thr_test, (proba >= 0.5).astype(int), proba)
    bands = profiles.loc[test_rows, "Predicted_Risk_Level"].tolist()
    band_metrics = evaluation.band_breakdown(scores_test, rel_test, bands, (5, 10))
    print(f"      P@5={metrics['precision@5']} R@10={metrics['recall@10']} NDCG@10={metrics['ndcg@10']} R2={regression['r2']}")

    step("Strategy ablation (single-signal baselines vs the hybrid model)")
    pop = np.tile(np.array([centrality[r] for r in resources["Resource_ID"]])[None, :], (len(test_rows), 1))
    linear = (0.35 * M["rule_score"] + 0.25 * M["content_score"] + 0.20 * M["graph_score"]
              + 0.12 * M["style_type_fit"] + 0.08 * M["difficulty_fit"])[test_rows]
    strategies = {
        "Popularity / centrality baseline": pop,
        "Rule-based filtering only": M["rule_score"][test_rows],
        "Content-based only (TF-IDF)": M["content_score"][test_rows],
        "Knowledge graph + GNN only": M["graph_score"][test_rows],
        "Hybrid - fixed linear weights": linear,
        "Hybrid - Random Forest (proposed)": scores_test,
    }
    ablation = evaluation.strategy_ablation(strategies, rel_test, (5, 10))

    step("Benchmarking alternative ranking models")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    sub = rng.choice(len(X_train), size=min(45000, len(X_train)), replace=False)
    Xs, ys = X_train[sub], y_train[sub]
    bench = {
        "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=13, min_samples_leaf=20,
                                               n_jobs=-1, random_state=config.RANDOM_SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                                       random_state=config.RANDOM_SEED),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=25,
                                               random_state=config.RANDOM_SEED),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    model_comparison = []
    for name, mdl in bench.items():
        t = time.time()
        mdl.fit(Xs, ys)
        secs = round(time.time() - t, 2)
        pr = mdl.predict(X_test)
        rm = evaluation.regression_metrics(y_test, pr)
        rk = evaluation.ranking_metrics(pr.reshape(len(test_rows), R), rel_test, (5, 10))
        model_comparison.append({"model": name, "r2": rm["r2"], "rmse": rm["rmse"],
                                 "precision@5": rk["precision@5"], "ndcg@10": rk["ndcg@10"],
                                 "fit_seconds": secs})
        print(f"      {name:<20} R2={rm['r2']:.4f} NDCG@10={rk['ndcg@10']:.4f} ({secs}s)")

    step("Explainability - global importance, permutation importance, additive contributions")
    vsub = rng.choice(len(X_test), size=min(8000, len(X_test)), replace=False)
    importance = explainability.global_importance(model, FEATURE_NAMES, X_test[vsub], y_test[vsub])

    step("Scoring the full cohort and ranking the catalogue for every learner")
    full_scores = recommender.score_matrix(model, M, None, R)
    order = np.argsort(-full_scores, axis=1)
    top_idx = order[:, :config.TOP_K]

    step("Local explanations for every learner (SHAP-style additive tree contributions)")
    rows_top = np.arange(S) * R + top_idx[:, 0]
    X_all_top = np.empty((S, len(FEATURE_NAMES)), dtype=np.float32)
    for j, name in enumerate(FEATURE_NAMES):
        X_all_top[:, j] = M[name][np.arange(S), top_idx[:, 0]]
    contrib, bias = explainability.tree_contributions(model, X_all_top, len(FEATURE_NAMES))
    mean_abs = np.abs(contrib).mean(axis=0)
    keep = np.argsort(-mean_abs)[:18]
    contrib_summary = {"features": [FEATURE_NAMES[i] for i in keep],
                       "mean": [float(round(contrib[:, i].mean(), 5)) for i in keep],
                       "mean_abs": [float(round(mean_abs[i], 5)) for i in keep],
                       "bias": float(round(bias, 5))}
    lime_sample = {}
    for i in rng.choice(S, size=40, replace=False):
        lime_sample[profiles.loc[i, "student_id"]] = explainability.lime_explain(
            model, X_all_top[i], X_test[:4000], FEATURE_NAMES)

    step("Generating recommendations, interventions and per-learner payloads")
    res_records = []
    for _, r in resources.iterrows():
        res_records.append({
            "id": r["Resource_ID"], "title": r["Resource_Title"], "lesson": r["Lesson_Name"],
            "topic": r["Topic"], "sub": r["Sub_Topic"], "type": r["Resource_Type"],
            "level": r["Difficulty_Level"], "minutes": int(r["Estimated_Duration_Min"]),
            "week": r["Week"], "url": r["LMS_URL"], "tags": r["tag_list"],
            "prereq": r["prereq_list"], "centrality": round(float(centrality[r["Resource_ID"]]), 4)})
    res_by_pos = {i: res_records[i] for i in range(R)}

    def strat_label(i, j):
        """Which signal family did the heaviest lifting for this pair."""
        if M["lesson_gap"][i, j] > 0.5:
            return "Rule-based filter"
        cands = {
            "Knowledge graph + GNN": max(M["graph_score"][i, j], M["kg_proximity"][i, j]),
            "Content-based (TF-IDF)": max(M["content_score"][i, j], M["tag_overlap"][i, j]),
            "Context-aware": 0.7 * M["style_type_fit"][i, j] + 0.3 * M["difficulty_fit"][i, j],
            "Rule-based filter": M["prereq_readiness"][i, j] * M["duration_fit"][i, j],
        }
        return max(cands, key=cands.get)

    students_payload, index_rows, rec_rows, action_rows = {}, [], [], []
    recs_for_kg = {}
    eng_rank = profiles["engagement_score"].rank(pct=True)
    perf_rank = profiles["performance_score"].rank(pct=True)
    for i in range(S):
        row = profiles.loc[i]
        sid = row["student_id"]
        recs = []
        for rank, j in enumerate(top_idx[i], start=1):
            rr = res_by_pos[j]
            recs.append({"r": rr["id"], "rel": round(float(full_scores[i, j]), 4), "rank": rank,
                         "why": recommender.explain_pair(M, i, j), "strat": strat_label(i, j),
                         "gap": bool(M["lesson_gap"][i, j] > 0.5)})
            rec_rows.append({"student_id": sid, "rank": rank, "resource_id": rr["id"],
                             "resource_title": rr["title"], "lesson": rr["lesson"], "type": rr["type"],
                             "difficulty": rr["level"], "minutes": rr["minutes"],
                             "predicted_relevance": round(float(full_scores[i, j]), 4),
                             "strategy": recs[-1]["strat"], "reason": recs[-1]["why"],
                             "risk_level": row["Predicted_Risk_Level"], "learning_style": row["Learning_Style"],
                             "lms_url": rr["url"]})
        recs_for_kg[sid] = [{"resource_id": r["r"]} for r in recs]
        top_res = [{"title": res_by_pos[j]["title"], "type": res_by_pos[j]["type"],
                    "minutes": res_by_pos[j]["minutes"], "level": res_by_pos[j]["level"],
                    "lesson": res_by_pos[j]["lesson"]} for j in top_idx[i]]
        actions = intervention_mapper.build_actions(row, top_res)
        for a in actions:
            action_rows.append({"student_id": sid, "risk_level": row["Predicted_Risk_Level"], **a})
        impact = explainability.top_contributions(contrib[i], FEATURE_NAMES, PRETTY, 8)
        students_payload[sid] = {
            "id": sid, "name": row["Name"], "gender": row["gender"], "program": row["program"],
            "gpa": float(row["gpa"]), "priorResults": row["previous_results"], "grade": row["final_grade"],
            "course": row["course_information"], "moduleId": row["module_id"],
            "totalMarks": float(row["total_marks"]), "caMarks": float(row["ca_marks"]),
            "finalExamMarks": float(row["final_exam_marks"]), "examResults": float(row["exam_results"]),
            "quizScores": float(row["quiz_scores"]), "assignmentMarks": float(row["assignment_marks"]),
            "attendancePct": float(row["attendance_percentage"]),
            "sessionsAttended": int(row["sessions_attended"]), "sessionsHeld": int(row["total_sessions_held"]),
            "loginFrequency": int(row["login_frequency"]), "timeSpentLms": float(row["time_spent_lms"]),
            "resourcesAccessed": int(row["resources_accessed"]), "forumActivity": int(row["forum_chat_activity"]),
            "clickCount": int(row["click_count"]), "submissionStatus": row["submission_status"],
            "delayDays": int(row["submission_delay_days"]), "assessmentType": row["assessment_type"],
            "assessmentScore": float(row["assessment_score"]), "assessmentMax": float(row["assessment_max_score"]),
            "passFail": row["pass_fail_status"], "riskLevel": row["Predicted_Risk_Level"],
            "riskScore": round(float(row["risk_score"]), 3), "segment": row["Learning_Style"],
            "engagement": float(row["engagement_score"]), "performance": float(row["performance_score"]),
            "ability": round(float(row["ability"]), 3), "priority": float(row["intervention_priority"]),
            "gapLessons": list(row["gap_lessons"]),
            "engagementPercentile": round(float(eng_rank.iloc[i]) * 100, 1),
            "performancePercentile": round(float(perf_rank.iloc[i]) * 100, 1),
            "recommendations": recs, "actions": actions, "features": impact,
            "explanationBias": contrib_summary["bias"],
            "inTrainingSet": bool(i in set(train_rows.tolist())),
        }
        index_rows.append({"id": sid, "name": row["Name"], "program": row["program"],
                           "riskLevel": row["Predicted_Risk_Level"], "segment": row["Learning_Style"],
                           "engagement": float(row["engagement_score"]),
                           "performance": float(row["performance_score"]),
                           "gaps": int(row["n_gap_lessons"]), "shard": i // 250})

    step("Aggregating dashboard analytics")
    fam = {}
    for f, v in zip(FEATURE_NAMES, model.feature_importances_):
        fam[recommender.strategy_family(f)] = fam.get(recommender.strategy_family(f), 0.0) + float(v)
    tot = sum(fam.values()) or 1.0
    strategy_mix = [{"name": k, "value": round(v / tot * 100, 1)} for k, v in
                    sorted(fam.items(), key=lambda kv: -kv[1])]
    counts = np.bincount(top_idx.reshape(-1), minlength=R)
    coverage_by_type = {}
    for t_ in resources["Resource_Type"].unique():
        pos = np.where(resources["Resource_Type"].to_numpy() == t_)[0]
        coverage_by_type[t_] = {"recommended": int(counts[pos].sum()),
                                "available": round(float(len(pos) / R), 4), "resources": int(len(pos))}
    cat_counts, band_counts = {}, {b: {} for b in ["Low Risk", "Medium Risk", "High Risk"]}
    for a in action_rows:
        cat_counts[a["category"]] = cat_counts.get(a["category"], 0) + 1
        band_counts[a["risk_level"]][a["category"]] = band_counts[a["risk_level"]].get(a["category"], 0) + 1
    styles = sorted(profiles["Learning_Style"].unique())
    levels = ["Foundational", "Beginner", "Intermediate", "Advanced"]
    lv = resources["Difficulty_Level"].to_numpy()
    hm = []
    for st in styles:
        rows_ = np.where(profiles["Learning_Style"].to_numpy() == st)[0]
        picks = lv[top_idx[rows_].reshape(-1)]
        hm.append([round(float((picks == l).mean()), 4) for l in levels])
    topk_curve = [{"k": k, "precision": metrics[f"precision@{k}"], "recall": metrics[f"recall@{k}"],
                   "ndcg": metrics[f"ndcg@{k}"], "map": metrics[f"map@{k}"]} for k in config.K_VALUES]

    step("Writing outputs/")
    out = config.OUTPUT_DIR
    joblib.dump(model, out / "recommender_rf.joblib", compress=3)
    joblib.dump(clf, out / "relevance_classifier.joblib", compress=3)
    joblib.dump(nlp["vectorizer"], out / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump({"nodes": gnn["nodes"], "embeddings": gnn["embeddings"].astype(np.float32),
                 "history": gnn["history"]}, out / "gnn_embeddings.joblib", compress=3)
    joblib.dump(G, out / "knowledge_graph.joblib", compress=3)
    pd.DataFrame(rec_rows).to_csv(out / "recommendations.csv", index=False)
    pd.DataFrame(action_rows).to_csv(out / "interventions.csv", index=False)
    prof_cols = ["student_id", "Name", "program", "gender", "gpa", "final_grade", "total_marks",
                 "attendance_percentage", "quiz_scores", "login_frequency", "time_spent_lms",
                 "resources_accessed", "forum_chat_activity", "click_count", "engagement_score",
                 "performance_score", "ability", "risk_score", "Predicted_Risk_Level",
                 "Learning_Style", "n_gap_lessons", "Non_Engaged_Modules", "intervention_priority"]
    profiles[prof_cols].to_csv(out / "student_profiles.csv", index=False)
    pd.DataFrame({"feature": FEATURE_NAMES, "label": [PRETTY[f] for f in FEATURE_NAMES],
                  "impurity_importance": model.feature_importances_,
                  "permutation_importance": [importance["permutation"][f] for f in FEATURE_NAMES],
                  "mean_contribution": contrib.mean(axis=0),
                  "mean_abs_contribution": mean_abs,
                  "strategy_family": [recommender.strategy_family(f) for f in FEATURE_NAMES]}
                 ).sort_values("impurity_importance", ascending=False).to_csv(
        out / "feature_importance.csv", index=False)
    pd.DataFrame(contrib, columns=FEATURE_NAMES).assign(student_id=profiles["student_id"]).to_csv(
        out / "global_explainability.csv", index=False)
    heldout = pd.DataFrame({"student_id": np.repeat(profiles.loc[test_rows, "student_id"].to_numpy(), R),
                            "resource_id": np.tile(resources["Resource_ID"].to_numpy(), len(test_rows)),
                            "actual_relevance": y_test.round(4),
                            "predicted_relevance": y_pred_test.round(4),
                            "is_relevant": thr_test.astype(int)})
    heldout.to_csv(out / "heldout_pair_scores.csv", index=False)
    pd.DataFrame(ablation).to_csv(out / "strategy_comparison.csv", index=False)
    pd.DataFrame(model_comparison).to_csv(out / "model_comparison.csv", index=False)
    pd.DataFrame([{"student_id": k, **v} for k, v in lime_sample.items()]).to_csv(
        out / "lime_local_explanations.csv", index=False)

    all_metrics = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "learners": S, "resources": R, "pairs": S * R,
                   "train_learners": int(len(train_rows)), "holdout_learners": int(len(test_rows)),
                   "ranking": metrics, "regression": regression, "classification": classification,
                   "catalogue": catalogue, "per_risk_band": band_metrics, "ablation": ablation,
                   "model_comparison": model_comparison, "topk_curve": topk_curve,
                   "knowledge_graph": kg_stats, "gnn": {"dim": config.GNN_DIM,
                                                        "epochs": config.GNN_EPOCHS,
                                                        "final_link_auc": gnn["history"][-1]["val_auc"],
                                                        "final_loss": gnn["history"][-1]["loss"]},
                   "strategy_mix": strategy_mix, "feature_importance": importance}
    (out / "evaluation_metrics.json").write_text(json.dumps(all_metrics, indent=2))

    step("Rendering figures and the HTML report windows")
    ctx = {"profiles": profiles, "resources": resources, "report": report,
           "normalisation": (norm_before, norm_after), "importance": importance,
           "topk_curve": topk_curve, "regression": regression, "classification": classification,
           "metrics": metrics, "catalogue": catalogue, "ablation": ablation,
           "model_comparison": model_comparison, "band_metrics": band_metrics,
           "coverage_by_type": coverage_by_type, "intervention_categories": cat_counts,
           "intervention_by_risk": band_counts, "contrib_summary": contrib_summary,
           "graph": G, "kg_stats": kg_stats, "gnn_history": gnn["history"],
           "sample_graph": knowledge_graph.sample_subgraph(G, profiles),
           "style_difficulty": {"styles": styles, "levels": levels, "matrix": hm},
           "scatter": {"y_true": y_test[::9], "y_pred": y_pred_test[::9]},
           "n_students": S, "n_resources": R, "n_pairs": S * R}
    figures = visualization.generate_all(ctx)
    visualization.write_visual_report(ctx, figures)
    sample_ids = [n for n in ctx["sample_graph"].nodes if str(n).startswith("ST")]
    payload = kg_view.build_payload(profiles, resources, G, recs_for_kg, sample_ids)
    kg_view.write(payload, kg_stats)
    print(f"      {len(figures)} figures written")

    step("Publishing dashboard data + reports into the React app (frontend/public)")
    meta = {
        "generatedAt": all_metrics["generated"], "nStudents": S, "nResources": R, "nPairs": S * R,
        "topK": config.TOP_K, "trainLearners": int(len(train_rows)), "holdoutLearners": int(len(test_rows)),
        "metrics": metrics, "regression": regression, "classification": classification,
        "catalogue": {k: v for k, v in catalogue.items() if k != "resource_hit_counts"},
        "bandMetrics": band_metrics, "ablation": ablation, "modelComparison": model_comparison,
        "topkCurve": topk_curve, "strategyMix": strategy_mix,
        "gnn": all_metrics["gnn"], "kg": kg_stats, "preprocessing": report,
        "resources": res_records,
        "resourceUsage": [{"id": res_records[i]["id"], "title": res_records[i]["title"],
                           "lesson": res_records[i]["lesson"], "type": res_records[i]["type"],
                           "count": int(counts[i])} for i in range(R)],
        "coverageByType": coverage_by_type,
        "featureImportance": [{"feature": f, "label": PRETTY[f],
                               "value": round(float(v), 5),
                               "family": recommender.strategy_family(f)}
                              for f, v in sorted(zip(FEATURE_NAMES, model.feature_importances_),
                                                 key=lambda kv: -kv[1])[:14]],
        "globalImpact": [{"label": PRETTY[f], "impact": m} for f, m in
                         zip(contrib_summary["features"], contrib_summary["mean"])],
        "interventionCategories": cat_counts, "interventionByRisk": band_counts,
        "styleDifficulty": {"styles": styles, "levels": levels, "matrix": hm},
        "figures": figures,
        "lessons": [{"name": l, "resources": int((resources["Lesson_Name"] == l).sum()),
                     "gapLearners": int(sum(1 for g in profiles["gap_lessons"] if l in g))}
                    for l in sorted(resources["Lesson_Name"].unique())],
        "module1": {
            "styleDistribution": profiles["Learning_Style"].value_counts().to_dict(),
            "programDistribution": profiles["program"].value_counts().to_dict(),
            "genderDistribution": profiles["gender"].value_counts().to_dict(),
            "priorDistribution": profiles["previous_results"].value_counts().to_dict(),
            "segmentProfile": [
                {"segment": st,
                 "learners": int((profiles["Learning_Style"] == st).sum()),
                 "engagement": round(float(profiles.loc[profiles["Learning_Style"] == st, "engagement_score"].mean()), 1),
                 "performance": round(float(profiles.loc[profiles["Learning_Style"] == st, "performance_score"].mean()), 1),
                 "attendance": round(float(profiles.loc[profiles["Learning_Style"] == st, "attendance_percentage"].mean()), 1),
                 "logins": round(float(profiles.loc[profiles["Learning_Style"] == st, "login_frequency"].mean()), 1),
                 "lmsTime": round(float(profiles.loc[profiles["Learning_Style"] == st, "time_spent_lms"].mean()), 1),
                 "gaps": round(float(profiles.loc[profiles["Learning_Style"] == st, "n_gap_lessons"].mean()), 2)}
                for st in styles],
            "scatter": [{"x": float(profiles.loc[i, "engagement_score"]),
                         "y": float(profiles.loc[i, "performance_score"]),
                         "s": profiles.loc[i, "Learning_Style"],
                         "r": profiles.loc[i, "Predicted_Risk_Level"]}
                        for i in range(0, S, max(1, S // 600))],
        },
        "module2": {
            "riskDistribution": profiles["Predicted_Risk_Level"].value_counts().to_dict(),
            "gradeDistribution": profiles["final_grade"].value_counts().to_dict(),
            "passFail": profiles["pass_fail_status"].value_counts().to_dict(),
            "riskByStyle": [{"segment": st,
                             **{b: int(((profiles["Learning_Style"] == st) &
                                        (profiles["Predicted_Risk_Level"] == b)).sum())
                                for b in ["Low Risk", "Medium Risk", "High Risk"]}} for st in styles],
            "featureMeans": [{"band": b,
                              "totalMarks": round(float(profiles.loc[profiles["Predicted_Risk_Level"] == b, "total_marks"].mean()), 1),
                              "attendance": round(float(profiles.loc[profiles["Predicted_Risk_Level"] == b, "attendance_percentage"].mean()), 1),
                              "quiz": round(float(profiles.loc[profiles["Predicted_Risk_Level"] == b, "quiz_scores"].mean()), 1),
                              "engagement": round(float(profiles.loc[profiles["Predicted_Risk_Level"] == b, "engagement_score"].mean()), 1),
                              "performance": round(float(profiles.loc[profiles["Predicted_Risk_Level"] == b, "performance_score"].mean()), 1),
                              "learners": int((profiles["Predicted_Risk_Level"] == b).sum())}
                             for b in ["Low Risk", "Medium Risk", "High Risk"]],
        },
    }
    (out / "dashboard_data.json").write_text(json.dumps(meta, separators=(",", ":")))
    (out / "students_payload.json").write_text(json.dumps(students_payload, separators=(",", ":")))
    (out / "students_index.json").write_text(json.dumps(index_rows, separators=(",", ":")))
    pd.DataFrame(X_all_top, columns=FEATURE_NAMES).assign(
        student_id=profiles["student_id"],
        resource_id=[resources["Resource_ID"].iloc[j] for j in top_idx[:, 0]],
        predicted_relevance=[round(float(full_scores[i, top_idx[i, 0]]), 5) for i in range(S)]
    ).to_csv(out / "top1_features.csv", index=False)

    fdata = config.FRONTEND_DATA_DIR
    (fdata / "students").mkdir(parents=True, exist_ok=True)
    config.FRONTEND_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (fdata / "meta.json").write_text(json.dumps(meta, separators=(",", ":")))
    (fdata / "students_index.json").write_text(json.dumps(index_rows, separators=(",", ":")))
    shards = {}
    for r_ in index_rows:
        shards.setdefault(r_["shard"], {})[r_["id"]] = students_payload[r_["id"]]
    for sh, blob in shards.items():
        (fdata / "students" / f"shard_{sh:02d}.json").write_text(json.dumps(blob, separators=(",", ":")))
    for name in ("visual_report.html", "knowledge_graph.html", "evaluation_metrics.json"):
        shutil.copy(out / name, config.FRONTEND_REPORT_DIR / name)
    figdst = config.FRONTEND_REPORT_DIR / "figures"
    figdst.mkdir(exist_ok=True)
    for f in figures:
        shutil.copy(config.FIGURE_DIR / f["file"], figdst / f["file"])

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"  outputs           -> {out}")
    print(f"  react data        -> {fdata}")
    print(f"  report window     -> {config.FRONTEND_REPORT_DIR/'visual_report.html'}")
    print(f"  knowledge graph   -> {config.FRONTEND_REPORT_DIR/'knowledge_graph.html'}")
    print(f"  Precision@5={metrics['precision@5']}  Recall@10={metrics['recall@10']}  "
          f"NDCG@10={metrics['ndcg@10']}  MAP@10={metrics['map@10']}  R2={regression['r2']}")


if __name__ == "__main__":
    main()
