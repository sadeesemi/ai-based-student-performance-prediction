"""CLI demo:  python recommend_student.py ST1008  [--verify]

Prints the trained recommendation output for one learner. With --verify the saved
Random Forest is reloaded and re-scores the learner's top ranked pair, proving the
dashboard numbers come from the trained model.
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "recommendation_module"

from .service import ArtefactsMissing, service  # noqa: E402
from .src import config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student_id")
    ap.add_argument("--verify", action="store_true", help="reload the model and re-score")
    args = ap.parse_args()

    try:
        s = service.get_student(args.student_id)
    except ArtefactsMissing as err:
        print(err)
        return 1
    except KeyError as err:
        print(err)
        return 1

    res = {r["id"]: r for r in service.resources()}
    print("=" * 96)
    print(f"{s['name']}  ({s['id']})   {s['program']} - {s['course']}")
    print(f"  Module 01 segment : {s['segment']}")
    print(f"  Module 02 risk    : {s['riskLevel']}  (risk score {s['riskScore']})")
    print(f"  engagement {s['engagement']}/100   performance {s['performance']}/100   "
          f"attendance {s['attendancePct']}%   GPA {s['gpa']}")
    print(f"  non-engaged lessons: {', '.join(s['gapLessons']) or 'none'}")
    print("-" * 96)
    print("TOP RECOMMENDED RESOURCES")
    for r in s["recommendations"]:
        meta = res.get(r["r"], {})
        print(f"  {r['rank']}. [{r['rel']:.3f}] {meta.get('title', r['r'])}")
        print(f"       {meta.get('lesson')} | {meta.get('type')} | {meta.get('minutes')} min | "
              f"{meta.get('level')} | signal: {r['strat']}")
        print(f"       why: {r['why']}")
        print(f"       {meta.get('url')}")
    print("-" * 96)
    print("GENERATED INTERVENTIONS")
    for a in s["actions"]:
        print(f"  [{a['priority']:<8}] {a['title']}  ({a['category']} - {a['status']})")
        print(f"             {a['detail']}")
    print("-" * 96)
    print("EXPLANATION - additive contributions to the top ranked resource")
    for f in s["features"]:
        bar = "#" * max(1, int(abs(f["impact"]) * 260))
        print(f"  {f['label'][:44]:<46} {f['impact']:+.5f} {bar}")

    if args.verify:
        import joblib
        import pandas as pd
        model = joblib.load(config.OUTPUT_DIR / "recommender_rf.joblib")
        df = pd.read_csv(config.OUTPUT_DIR / "top1_features.csv")
        row = df[df["student_id"] == s["id"]]
        feats = [c for c in df.columns if c not in ("student_id", "resource_id", "predicted_relevance")]
        pred = float(model.predict(row[feats].to_numpy())[0])
        print("-" * 96)
        print(f"VERIFY  stored={float(row['predicted_relevance'].iloc[0]):.5f}  "
              f"model re-prediction={pred:.5f}  resource={row['resource_id'].iloc[0]}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
