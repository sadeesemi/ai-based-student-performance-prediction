"""Data ingestion layer (Phase 1.1 - Data Integration & Ingestion).

Two sources are consumed:
  * student_dataset.csv  - carries the Module 01 profiling output (Learning_Style)
                           and the Module 02 prediction output (Predicted_Risk_Level)
  * resources.csv        - the indexed LMS resource catalogue
"""
import unicodedata
import re
import pandas as pd

from . import config

_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_csv(path):
    last = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as exc:  # pragma: no cover - encoding fallback
            last = exc
    raise last


def _tidy(df):
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(_clean_text)
    return df.drop_duplicates()


def _clean_text(value):
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize("NFKD", value)
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    return re.sub(r"\s+", " ", value).strip()


def load_students(path=None):
    df = _tidy(_read_csv(path or config.STUDENT_FILE))
    df = df.drop_duplicates(subset=["student_id"], keep="first").reset_index(drop=True)
    return df


def load_resources(path=None):
    df = _tidy(_read_csv(path or config.RESOURCE_FILE))
    df = df.drop_duplicates(subset=["Resource_ID"], keep="first").reset_index(drop=True)
    df["Prerequisite_Resource_IDs"] = df["Prerequisite_Resource_IDs"].fillna("")
    df["tag_list"] = df["Knowledge_Graph_Tag"].fillna("").map(
        lambda v: [t.strip() for t in str(v).split(",") if t.strip()])
    df["prereq_list"] = df["Prerequisite_Resource_IDs"].map(
        lambda v: [t.strip() for t in str(v).split(",") if t.strip()])
    return df


def load_all():
    return load_students(), load_resources()
