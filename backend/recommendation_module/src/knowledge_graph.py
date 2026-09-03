"""Phase 1.4 - Knowledge graph construction.

Heterogeneous graph over Student / Lesson / Topic / Resource / Tag /
Risk-Level / Learning-Style nodes.  Week nodes are intentionally excluded:
the learner dataset carries no week-level signal, so a week entity would add
edges that cannot be grounded in the data.
"""
import numpy as np

try:  # networkx when installed, bundled fallback otherwise
    import networkx as nx
except ImportError:  # pragma: no cover
    from . import minigraph as nx

from . import config

NODE_TYPES = ["student", "lesson", "topic", "resource", "tag", "risk", "style"]
EDGE_TYPES = ["NOT_ENGAGED_IN", "HAS_RISK", "LEARNS_AS", "COVERS", "PART_OF",
              "HAS_RESOURCE", "TAGGED_AS", "REQUIRES", "RELATED_TO"]


def build(profiles, resources):
    G = nx.Graph()

    # ---- catalogue side ----------------------------------------------------
    for _, r in resources.iterrows():
        rid = r["Resource_ID"]
        G.add_node(rid, kind="resource", label=r["Resource_Title"], lesson=r["Lesson_Name"],
                   topic=r["Topic"], rtype=r["Resource_Type"], level=r["Difficulty_Level"],
                   minutes=int(r["Estimated_Duration_Min"]), url=r["LMS_URL"])
        lesson_node = f"L::{r['Lesson_Name']}"
        topic_node = f"T::{r['Topic']}"
        G.add_node(lesson_node, kind="lesson", label=r["Lesson_Name"])
        G.add_node(topic_node, kind="topic", label=r["Topic"])
        G.add_edge(lesson_node, topic_node, rel="COVERS", weight=1.0)
        G.add_edge(topic_node, rid, rel="HAS_RESOURCE", weight=1.0)
        G.add_edge(lesson_node, rid, rel="PART_OF", weight=1.0)
        for tag in r["tag_list"]:
            tag_node = f"K::{tag}"
            G.add_node(tag_node, kind="tag", label=tag)
            G.add_edge(rid, tag_node, rel="TAGGED_AS", weight=0.6)

    for _, r in resources.iterrows():
        for pre in r["prereq_list"]:
            if pre in G:
                G.add_edge(r["Resource_ID"], pre, rel="REQUIRES", weight=0.9)

    # ---- learner side ------------------------------------------------------
    for _, s in profiles.iterrows():
        sid = s["student_id"]
        G.add_node(sid, kind="student", label=s["Name"], risk=s["Predicted_Risk_Level"],
                   style=s["Learning_Style"], engagement=float(s["engagement_score"]),
                   performance=float(s["performance_score"]), program=s["program"])
        risk_node = f"R::{s['Predicted_Risk_Level']}"
        style_node = f"S::{s['Learning_Style']}"
        G.add_node(risk_node, kind="risk", label=s["Predicted_Risk_Level"])
        G.add_node(style_node, kind="style", label=s["Learning_Style"])
        G.add_edge(sid, risk_node, rel="HAS_RISK", weight=float(s["risk_weight"]))
        G.add_edge(sid, style_node, rel="LEARNS_AS", weight=1.0)
        for lesson in s["gap_lessons"]:
            ln = f"L::{lesson}"
            if ln in G:
                G.add_edge(sid, ln, rel="NOT_ENGAGED_IN", weight=1.0)

    return G


def stats(G):
    kinds = {}
    for _, d in G.nodes(data=True):
        kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
    rels = {}
    for _, _, d in G.edges(data=True):
        rels[d.get("rel", "?")] = rels.get(d.get("rel", "?"), 0) + 1
    degrees = np.array([d for _, d in G.degree()])
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": kinds,
        "edge_types": rels,
        "avg_degree": float(round(degrees.mean(), 2)),
        "density": float(round(nx.density(G), 6)),
        "components": nx.number_connected_components(G),
    }


def resource_centrality(G, resources):
    """Degree centrality restricted to the catalogue sub-graph (KG signal)."""
    sub = G.subgraph([n for n, d in G.nodes(data=True) if d["kind"] != "student"])
    cent = nx.degree_centrality(sub)
    vals = np.array([cent.get(rid, 0.0) for rid in resources["Resource_ID"]])
    if vals.max() > 0:
        vals = vals / vals.max()
    return dict(zip(resources["Resource_ID"], vals))


def lesson_proximity(resources):
    """Meta-path proximity between lessons: 1 hop = shared topic/tag overlap.

    Returns lesson x lesson similarity in [0,1] used as the kg_proximity feature
    for learners whose gap lesson is not the resource's own lesson.
    """
    lessons = sorted(resources["Lesson_Name"].unique())
    tags = {l: set() for l in lessons}
    topics = {l: set() for l in lessons}
    for _, r in resources.iterrows():
        tags[r["Lesson_Name"]].update(r["tag_list"])
        topics[r["Lesson_Name"]].add(r["Topic"])
    prox = {}
    for a in lessons:
        for b in lessons:
            if a == b:
                prox[(a, b)] = 1.0
                continue
            jt = len(tags[a] & tags[b]) / max(len(tags[a] | tags[b]), 1)
            jp = len(topics[a] & topics[b]) / max(len(topics[a] | topics[b]), 1)
            prox[(a, b)] = round(0.65 * jt + 0.35 * jp, 4)
    return prox


def sample_subgraph(G, profiles, n_students=None, seed=config.RANDOM_SEED):
    """A readable slice of the live graph for the interactive HTML view."""
    n_students = n_students or config.KG_SAMPLE_STUDENTS
    rng = np.random.default_rng(seed)
    per_risk = max(1, n_students // 3)
    picked = []
    for level in ["High Risk", "Medium Risk", "Low Risk"]:
        pool = profiles.loc[profiles["Predicted_Risk_Level"] == level, "student_id"].to_numpy()
        if len(pool):
            picked += list(rng.choice(pool, size=min(per_risk, len(pool)), replace=False))
    keep = set(picked) | {n for n, d in G.nodes(data=True) if d["kind"] != "student"}
    return G.subgraph(keep).copy()
