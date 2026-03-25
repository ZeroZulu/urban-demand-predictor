"""
Standalone trainer — no MLflow, handles missing lag columns.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import joblib
import sqlalchemy
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

load_dotenv()

db_url = os.getenv("DATABASE_URL", "postgresql://analyst:localdev@localhost:5432/urban_demand")
engine = sqlalchemy.create_engine(db_url)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading ml_features from Postgres...")
df = pd.read_sql("SELECT * FROM ml_features ORDER BY hour_bucket, zone_id", engine)
df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])

# Show actual date range in data
print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")
print(f"Date range in data: {df['hour_bucket'].min().date()} → {df['hour_bucket'].max().date()}")
print(f"Columns: {list(df.columns)}")

# Add missing lag columns (not in our simplified view — fill with zone rolling mean)
print("\nComputing lag features from data...")
df = df.sort_values(["zone_id", "hour_bucket"]).reset_index(drop=True)
df["demand_lag_24h"]  = df.groupby("zone_id")["trip_count"].shift(24).fillna(df["trip_count"].mean())
df["demand_lag_168h"] = df.groupby("zone_id")["trip_count"].shift(168).fillna(df["trip_count"].mean())
df["demand_rolling_7d_avg"] = (
    df.groupby("zone_id")["trip_count"]
    .transform(lambda x: x.rolling(168, min_periods=1).mean())
)
print("Lag features added.")

# ── Temporal split on actual data range ───────────────────────────────────────
# Filter to only rows within a sensible date range (exclude any bad future dates)
data_min = df["hour_bucket"].min()
data_max = df["hour_bucket"].max()
print(f"\nActual data range: {data_min.date()} → {data_max.date()}")

# Use last 2 months of actual data as test
cutoff = data_max - pd.DateOffset(months=2)
train_df = df[df["hour_bucket"] <= cutoff].copy()
test_df  = df[df["hour_bucket"] >  cutoff].copy()
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}  Cutoff: {cutoff.date()}")

if len(test_df) < 100:
    print("WARNING: Test set too small. Using last 20% of data instead.")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()
    print(f"New split — Train: {len(train_df):,}  Test: {len(test_df):,}")

# ── Feature engineering ───────────────────────────────────────────────────────
from src.features.builder import build_features

X_train = build_features(train_df)
y_train = train_df["trip_count"].values
X_test  = build_features(test_df)
y_test  = test_df["trip_count"].values

val_cut = int(len(X_train) * 0.9)
X_tr, y_tr   = X_train.iloc[:val_cut], y_train[:val_cut]
X_val, y_val = X_train.iloc[val_cut:], y_train[val_cut:]
print(f"\nFeatures: {len(X_train.columns)}")
print(f"Train: {len(X_tr):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ── Metrics ───────────────────────────────────────────────────────────────────
def mape(y_true, y_pred):
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "mape": mape(np.array(y_true), np.array(y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

# ── Naive baseline ────────────────────────────────────────────────────────────
baseline_lookup = train_df.groupby(["zone_id","hour_of_day","day_of_week"])["trip_count"].mean()
baseline_preds = test_df.apply(
    lambda r: baseline_lookup.get(
        (r["zone_id"], r["hour_of_day"], r["day_of_week"]),
        train_df["trip_count"].mean()), axis=1).values
results = {"Naive baseline": {**evaluate(y_test, baseline_preds), "train_s": 0}}

# ── Train models ──────────────────────────────────────────────────────────────
models_cfg = [
    ("Random Forest", RandomForestRegressor(
        n_estimators=300, max_depth=14, min_samples_leaf=3,
        n_jobs=-1, random_state=42)),
    ("XGBoost", XGBRegressor(
        n_estimators=600, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, verbosity=0)),
    ("LightGBM", LGBMRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1)),
]

trained = {}
for name, model in models_cfg:
    print(f"\nTraining {name}...")
    t0 = time.time()
    if name == "XGBoost":
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=30, verbose=100)
    else:
        model.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    preds = np.maximum(model.predict(X_test), 0)
    m = {**evaluate(y_test, preds), "train_s": round(elapsed, 1)}
    results[name] = m
    trained[name] = model
    print(f"  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  "
          f"MAPE={m['mape']:.1f}%  R²={m['r2']:.4f}  ({elapsed:.0f}s)")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  MODEL COMPARISON — Real NYC Data (28.8M trips)")
print("="*65)
print(f"  {'Model':<20} {'RMSE':>7} {'MAE':>7} {'MAPE':>8} {'R²':>8} {'Time':>7}")
print("  " + "-"*57)
for name, m in results.items():
    t = f"{m['train_s']}s" if m['train_s'] else "—"
    print(f"  {name:<20} {m['rmse']:>7.2f} {m['mae']:>7.2f} "
          f"{m['mape']:>7.1f}% {m['r2']:>8.4f} {t:>7}")
print("="*65)

# ── Save best model ───────────────────────────────────────────────────────────
best_name = min(
    [k for k in results if k != "Naive baseline"],
    key=lambda k: results[k]["rmse"])
print(f"\nBest model: {best_name}")

Path("models").mkdir(exist_ok=True)
joblib.dump(trained[best_name],      "models/best_model.pkl")
joblib.dump(list(X_train.columns),   "models/feature_names.pkl")

print("Building SHAP explainer...")
import shap
explainer = shap.TreeExplainer(trained[best_name])
joblib.dump(explainer, "models/shap_explainer.pkl")
joblib.dump({
    "best_model": best_name, "model_version": "1.0.0",
    "feature_names": list(X_train.columns),
    "metrics": results[best_name], "all_results": results,
}, "models/model_metadata.pkl")

print("\nSaved models:")
for f in Path("models").glob("*.pkl"):
    print(f"  {f.name}  ({f.stat().st_size/1e3:.0f} KB)")
print("\nDone! Restart the API: docker-compose restart api")