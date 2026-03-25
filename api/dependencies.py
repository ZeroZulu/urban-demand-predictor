"""
FastAPI dependency injection for model artifacts.
Loaded once at startup via lru_cache — zero overhead per request.
"""
from src.models.predictor import (
    load_explainer,
    load_feature_names,
    load_metadata,
    load_model,
)


def get_model():
    return load_model()


def get_explainer():
    return load_explainer()


def get_feature_names():
    return load_feature_names()


def get_metadata():
    return load_metadata()
