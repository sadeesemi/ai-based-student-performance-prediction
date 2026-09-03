"""Maps risk band + behaviour gaps to structured intervention actions.

Covers the five intervention categories required by the module scope:
study planning, revision support, time management, learning resources and
academic support alerts.
"""
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
CATEGORIES = ["Study planning", "Revision support", "Time management",
              "Learning resource", "Academic support"]


def _act(title, category, priority, status, detail):
    return {"title": title, "category": category, "priority": priority,
            "status": status, "detail": detail}


def build_actions(row, top_resources, max_actions=5):
    a = []
    risk = row["Predicted_Risk_Level"]
    gaps = row["gap_lessons"]

    if risk == "High Risk":
        a.append(_act("Schedule an academic support meeting this week", "Academic support",
                      "Critical", "Alert raised",
                      f"Module 02 flags high risk (score {row['risk_score']:.2f}) with performance at "
                      f"{row['performance_score']:.0f}/100."))
    elif risk == "Medium Risk":
        a.append(_act("Progress check-in with the module tutor", "Academic support", "Medium",
                      "Suggested", "Medium risk band - confirm the recovery plan is on track."))

    if gaps:
        a.append(_act(f"Complete the non-engaged lesson: {gaps[0]}", "Study planning",
                      "High" if risk != "Low Risk" else "Medium", "Pending",
                      "Lesson has no recorded engagement in the LMS activity log."
                      + (f" {len(gaps)} lessons flagged in total." if len(gaps) > 1 else "")))

    if row["attendance_ratio"] < 0.6:
        a.append(_act("Catch up on missed lectures using the recordings", "Revision support",
                      "High", "Pending",
                      f"Attendance at {row['attendance_percentage']:.0f}% "
                      f"({int(row['sessions_attended'])}/{int(row['total_sessions_held'])} sessions)."))

    if row["is_late"] or row["submission_delay_days"] > 0:
        a.append(_act("Rebuild the assignment timeline with weekly milestones", "Time management",
                      "Medium", "Suggested",
                      f"Last submission was {max(int(row['submission_delay_days']), 1)} day(s) past the due date."))

    if row["quiz_gap"] > 0.55:
        a.append(_act("Run timed quiz practice on the weakest topic", "Revision support",
                      "High" if risk == "High Risk" else "Medium", "Pending",
                      f"Quiz average of {row['quiz_scores']:.0f} sits below the cohort mid-point."))

    if row["engagement_score"] < 40:
        a.append(_act("Lift LMS activity to three study sessions per week", "Study planning",
                      "High", "Pending",
                      f"Engagement index {row['engagement_score']:.0f}/100 with "
                      f"{int(row['login_frequency'])} logins recorded."))

    if row["forum_activity"] < 0.2:
        a.append(_act("Join the module discussion forum for peer support", "Study planning",
                      "Low", "Suggested", "Almost no forum or chat interaction logged."))

    if top_resources:
        r = top_resources[0]
        a.append(_act(f"Work through: {r['title']}", "Learning resource",
                      "High" if risk != "Low Risk" else "Medium", "Recommended",
                      f"{r['type']} - {r['minutes']} min - {r['level']} - {r['lesson']}."))

    if risk == "Low Risk" and row["ability"] > 0.65:
        a.append(_act("Extend with advanced material to consolidate mastery", "Learning resource",
                      "Low", "Suggested", "Low risk with strong ability - enrichment path unlocked."))

    seen, out = set(), []
    for item in sorted(a, key=lambda x: PRIORITY_ORDER[x["priority"]]):
        if item["title"] in seen:
            continue
        seen.add(item["title"])
        out.append(item)
    return out[:max_actions]
