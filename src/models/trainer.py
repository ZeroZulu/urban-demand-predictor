"""
UrbanPulse — Model Training Pipeline
MLflow-tracked training of Random Forest, XGBoost, and LightGBM.
Best model + SHAP explainer saved to disk for API serving.

Usage:
    python -m src.models.trainer
"""
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
import yaml
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.features.builder import build_features
from src.models.evaluator import comparison_table, evaluate
from src.utils.db import get_engine
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_ml_features() -> pd.DataFrame:
    """Load the full ML feature set from the PostgreSQL materialized view."""
    engine = get_engine()
    logger.info("Loading ml_features from Postgres...")
    df = pd.read_sql(
        "SELECT * FROM ml_features ORDER BY hour_bucket, zone_id",
        engine,
    )
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])
    logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def temporal_split(df: pd.DataFrame, test_months: int = 2):
    """
    Split by time — NEVER shuffle time-series data.
    Returns (train_df, test_df).
    """
    cutoff = df["hour_bucket"].max() - pd.DateOffset(months=test_months)
    train = df[df["hour_bucket"] <= cutoff].copy()
    test  = df[df["hour_bucket"] >  cutoff].copy()
    logger.info(
        f"Temporal split | Train: {len(train):,} rows "
        f"| Test: {len(test):,} rows | Cutoff: {cutoff.date()}"
    )
    return train, test


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> dict:
    """Bayesian hyperparameter search for XGBoost via Optuna."""

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "tree_method": "hist",
            "verbosity": 0,
        }
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        preds = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_val - preds) ** 2)))
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        f"Optuna best XGBoost RMSE: {study.best_value:.4f} "
        f"(params: {study.best_params})"
    )
    return study.best_params


# ── Training ──────────────────────────────────────────────────────────────────

def run_training() -> dict:
    cfg = load_config()
    modeling_cfg = cfg["modeling"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # Load & split data
    raw = load_ml_features()
    train_df, test_df = temporal_split(raw, test_months=modeling_cfg["test_months"])

    # Build feature matrices
    X_train_full = build_features(train_df)
    y_train_full = train_df["trip_count"].values
    X_test        = build_features(test_df)
    y_test         = test_df["trip_count"].values

    feature_names = X_train_full.columns.tolist()

    # Validation split from tail of training set (temporal)
    val_cut = int(len(X_train_full) * (1 - modeling_cfg["val_fraction"]))
    X_tr, y_tr = X_train_full.iloc[:val_cut], y_train_full[:val_cut]
    X_val, y_val = X_train_full.iloc[val_cut:], y_train_full[val_cut:]
    logger.info(f"Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Tune XGBoost
    logger.info("Running Optuna hyperparameter search for XGBoost...")
    best_xgb_params = tune_xgboost(
        X_tr, y_tr, X_val, y_val,
        n_trials=modeling_cfg["optuna"]["n_trials"]
    )

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=modeling_cfg["random_state"],
        ),
        "XGBoost": XGBRegressor(
            **best_xgb_params,
            tree_method="hist",
            random_state=modeling_cfg["random_state"],
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=modeling_cfg["random_state"],
            verbose=-1,
        ),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):

            if name == "XGBoost":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=100,
                )
            else:
                model.fit(X_tr, y_tr)

            preds  = model.predict(X_test)
            metrics = evaluate(y_test, preds)
            results[name] = metrics

            # Log to MLflow
            mlflow.log_params({"model_type": name})
            mlflow.log_params(
                {k: v for k, v in (model.get_params() or {}).items()
                 if isinstance(v, (int, float, str, bool))}
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path=name.lower())

            logger.info(
                f"{name:15s} | RMSE: {metrics['rmse']:.2f} | "
                f"MAE: {metrics['mae']:.2f} | "
                f"MAPE: {metrics['mape']:.1f}% | "
                f"R²: {metrics['r2']:.3f}"
            )

    # ── Select best model ──────────────────────────────────────────────────────
    best_name  = min(results, key=lambda k: results[k]["rmse"])
    best_model = models[best_name]
    logger.info(f"\nBest model: {best_name} (RMSE: {results[best_name]['rmse']:.2f})")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(best_model,  MODEL_DIR / "best_model.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")
    logger.info("Saved best_model.pkl and feature_names.pkl")

    # SHAP explainer — pre-built for fast API responses
    logger.info("Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(best_model)
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")
    logger.info("Saved shap_explainer.pkl")

    # ── Save model metadata ────────────────────────────────────────────────────
    metadata = {
        "best_model":    best_name,
        "model_version": cfg["api"]["model_version"],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "metrics":       results[best_name],
        "all_results":   results,
    }
    joblib.dump(metadata, MODEL_DIR / "model_metadata.pkl")

    # ── Print comparison table ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON (holdout test set)")
    print("=" * 60)
    print(comparison_table(results).to_string())
    print("=" * 60)
    print(f"\n  Best: {best_name}")
    print(f"  MLflow UI: http://localhost:5000\n")

    return results


if __name__ == "__main__":
    run_training()
