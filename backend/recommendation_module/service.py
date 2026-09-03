"""Application-layer service for the recommendation module.

The Flask blueprint (routes.py) and the CLI (recommend_student.py) both go through
this class, so there is exactly one place that knows how trained artefacts are read.
"""
import json
from functools import lru_cache

from .src import config


class ArtefactsMissing(RuntimeError):
    pass


class RecommendationService:
    def __init__(self, output_dir=None):
        self.dir = output_dir or config.OUTPUT_DIR

    # ---- loading ---------------------------------------------------------
    def _read(self, name):
        path = self.dir / name
        if not path.exists():
            raise ArtefactsMissing(
                f"{path.name} not found. Train the module first:  python main.py")
        return json.loads(path.read_text())

    @property
    @lru_cache(maxsize=None)
    def meta(self):
        return self._read("dashboard_data.json")

    @property
    @lru_cache(maxsize=None)
    def payloads(self):
        return self._read("students_payload.json")

    @property
    @lru_cache(maxsize=None)
    def index(self):
        return self._read("students_index.json")

    @property
    @lru_cache(maxsize=None)
    def metrics(self):
        return self._read("evaluation_metrics.json")

    # ---- queries ---------------------------------------------------------
    def get_meta(self):
        return self.meta

    def get_index(self):
        return self.index

    def get_student(self, student_id):
        sid = str(student_id).strip().upper()
        student = self.payloads.get(sid)
        if student is None:
            raise KeyError(f"Student {sid} is not part of the trained cohort")
        return student

    def search(self, query, limit=8):
        q = str(query or "").strip().lower()
        if not q:
            return []
        starts = [s for s in self.index if s["id"].lower().startswith(q) or s["name"].lower().startswith(q)]
        if len(starts) >= limit:
            return starts[:limit]
        rest = [s for s in self.index
                if s not in starts and (q in s["id"].lower() or q in s["name"].lower())]
        return (starts + rest)[:limit]

    def get_metrics(self):
        return self.metrics

    def resources(self):
        return self.meta["resources"]

    def cohort_summary(self):
        m = self.meta
        return {"learners": m["nStudents"], "resources": m["nResources"], "pairs": m["nPairs"],
                "generated": m["generatedAt"], "ranking": m["metrics"], "regression": m["regression"],
                "knowledge_graph": m["kg"], "strategy_mix": m["strategyMix"]}


service = RecommendationService()
