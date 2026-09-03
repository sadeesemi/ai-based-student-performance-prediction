"""Shared backend configuration."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "module3-dev-secret")
    JSON_SORT_KEYS = False
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/student_performance")
    OUTPUT_DIR = BASE_DIR / "recommendation_module" / "outputs"
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
