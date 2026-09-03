"""NLP layer - text preprocessing, TF-IDF vectorisation and semantic similarity.

Resource documents are built from title + lesson + topic + sub-topic + knowledge
graph tags + delivery format.  A learner 'need document' is synthesised from the
Module 01 profile (learning style, behaviour gaps) and the Module 02 risk band,
then matched against the catalogue with cosine similarity.
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config

STOPWORDS = {"the", "a", "an", "of", "to", "and", "for", "in", "on", "with", "by",
             "is", "are", "as", "at", "or", "lesson", "session", "part"}


def normalise(text):
    text = str(text).lower()
    text = re.sub(r"[_/\\-]", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = [t for t in text.split() if t and t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def resource_documents(resources):
    docs = []
    for _, r in resources.iterrows():
        parts = [r["Resource_Title"], r["Lesson_Name"], r["Topic"], r["Sub_Topic"],
                 r["Resource_Type"], r["Difficulty_Level"], " ".join(r["tag_list"])]
        docs.append(normalise(" ".join(str(p) for p in parts)))
    return docs


def lesson_topic_index(resources):
    idx = {}
    for lesson, grp in resources.groupby("Lesson_Name"):
        terms = list(grp["Topic"]) + list(grp["Sub_Topic"]) + [t for tl in grp["tag_list"] for t in tl]
        idx[lesson] = normalise(lesson + " " + " ".join(map(str, terms)))
    return idx


def need_document(row, lesson_terms):
    """Learner need text (Phase 1.3 - context & need analysis)."""
    bits = []
    for lesson in row["gap_lessons"]:
        bits.append(lesson)
        bits.append(lesson_terms.get(lesson, ""))
    if not row["gap_lessons"]:
        bits.append("general course introduction consolidation mastery extension")
    bits.append(config.STYLE_KEYWORDS.get(row["Learning_Style"], ""))
    bits.append(config.RISK_KEYWORDS.get(row["Predicted_Risk_Level"], ""))
    if row["attendance_ratio"] < 0.5:
        bits.append("lecture recording missed session catch up attendance")
    if row["quiz_gap"] > 0.55:
        bits.append("quiz practice self test questions revision")
    if row["assessment_gap"] > 0.55:
        bits.append("assignment activity group work submission template")
    if row["is_late"]:
        bits.append("time management planning deadline scheduling estimation")
    if row["forum_activity"] < 0.25:
        bits.append("discussion collaboration group activity peer")
    if row["performance_score"] < 45:
        bits.append("worked example fundamentals definitions principles basics")
    if row["ability"] > 0.7:
        bits.append("advanced automation containers pipeline metrics estimation")
    return normalise(" ".join(map(str, bits)))


def fit(profiles, resources):
    docs = resource_documents(resources)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    R = vec.fit_transform(docs)
    lesson_terms = lesson_topic_index(resources)
    queries = [need_document(row, lesson_terms) for _, row in profiles.iterrows()]
    Q = vec.transform(queries)
    content_sim = cosine_similarity(Q, R)                       # (students x resources)
    item_sim = cosine_similarity(R, R)                          # semantic item-item
    tag_sets = [set(normalise(" ".join(t)).split()) for t in resources["tag_list"]]
    tag_overlap = np.zeros_like(content_sim)
    q_tokens = [set(q.split()) for q in queries]
    for j, tags in enumerate(tag_sets):
        if not tags:
            continue
        for i, qt in enumerate(q_tokens):
            inter = len(qt & tags)
            if inter:
                tag_overlap[i, j] = inter / len(qt | tags)
    return {"vectorizer": vec, "resource_matrix": R, "query_matrix": Q, "queries": queries,
            "content_sim": content_sim, "item_sim": item_sim, "tag_overlap": tag_overlap,
            "lesson_terms": lesson_terms, "documents": docs}
