"""Flask blueprint: /api/recommendation/*

Optional. The React app reads the published JSON by default, but flipping
REACT_APP_USE_BACKEND=true in frontend/.env makes it call these endpoints instead
with no component changes.
"""
from flask import Blueprint, jsonify, request

from .service import ArtefactsMissing, service

recommendation_bp = Blueprint("recommendation", __name__, url_prefix="/api/recommendation")


@recommendation_bp.errorhandler(ArtefactsMissing)
def _missing(err):
    return jsonify({"error": str(err)}), 503


@recommendation_bp.get("/health")
def health():
    try:
        return jsonify({"status": "ok", **service.cohort_summary()})
    except ArtefactsMissing as err:
        return jsonify({"status": "untrained", "error": str(err)}), 503


@recommendation_bp.get("/meta")
def meta():
    return jsonify(service.get_meta())


@recommendation_bp.get("/students")
def students():
    q = request.args.get("q")
    if q:
        return jsonify(service.search(q, int(request.args.get("limit", 8))))
    return jsonify(service.get_index())


@recommendation_bp.get("/student/<student_id>")
def student(student_id):
    try:
        return jsonify(service.get_student(student_id))
    except KeyError as err:
        return jsonify({"error": str(err)}), 404


@recommendation_bp.get("/student/<student_id>/recommendations")
def student_recs(student_id):
    try:
        s = service.get_student(student_id)
    except KeyError as err:
        return jsonify({"error": str(err)}), 404
    return jsonify({"student_id": s["id"], "riskLevel": s["riskLevel"],
                    "recommendations": s["recommendations"], "actions": s["actions"],
                    "explanation": s["features"]})


@recommendation_bp.get("/resources")
def resources():
    return jsonify(service.resources())


@recommendation_bp.get("/metrics")
def metrics():
    return jsonify(service.get_metrics())
