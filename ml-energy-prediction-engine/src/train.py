"""
Stage 6: Train the final production model and save the complete pipeline.

This script reruns the exact pipeline we already validated interactively
(preprocessing -> feature engineering -> chronological split -> encode ->
Gradient Boosting), then saves everything the API will need at inference
time: the fitted preprocessor, the fitted model, and a small feature
config describing what columns/order the model expects.
"""
import json
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import load_and_clean
from feature_engineering import add_features
from split import chronological_split

CATEGORICAL_FEATURES = ["Source", "Day_Name"]
NUMERIC_FEATURES = [
    "Start_Hour", "Day_of_Year", "Year",
    "Production_Lag_1", "Production_Lag_24", "Production_Lag_168",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "Production"

MODELS_DIR = "models"


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:20s} MAE={mae:8.2f}  RMSE={rmse:8.2f}  R2={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    clean_df = load_and_clean()
    featured_df = add_features(clean_df)
    train_df, val_df, test_df = chronological_split(featured_df)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET]

    # Fit the preprocessor ONLY on training data.
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ], sparse_threshold=0)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)
    X_test_enc = preprocessor.transform(X_test)

    # Naive baseline, for the record.
    print("\n--- Validation ---")
    evaluate("Naive Baseline", y_val, val_df["Production_Lag_1"])
    lr = LinearRegression().fit(X_train_enc, y_train)
    evaluate("Linear Regression", y_val, lr.predict(X_val_enc))

    # Final production model: Gradient Boosting.
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train_enc, y_train)
    evaluate("Gradient Boosting", y_val, model.predict(X_val_enc))

    print("\n--- FINAL TEST SET (one-time check) ---")
    evaluate("Naive Baseline", y_test, test_df["Production_Lag_1"])
    test_metrics = evaluate("Gradient Boosting", y_test, model.predict(X_test_enc))

    # --- Save the pipeline: model + preprocessor + feature config ---
    joblib.dump(model, f"{MODELS_DIR}/production_model.joblib")
    joblib.dump(preprocessor, f"{MODELS_DIR}/preprocessor.joblib")

    feature_config = {
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "all_features_in_order": ALL_FEATURES,
        "target": TARGET,
        "valid_sources": sorted(clean_df["Source"].unique().tolist()),
        "valid_day_names": sorted(clean_df["Day_Name"].unique().tolist()),
        "test_set_metrics": test_metrics,
    }
    with open(f"{MODELS_DIR}/feature_config.json", "w") as f:
        json.dump(feature_config, f, indent=2)

    print(f"\nSaved: {MODELS_DIR}/production_model.joblib")
    print(f"Saved: {MODELS_DIR}/preprocessor.joblib")
    print(f"Saved: {MODELS_DIR}/feature_config.json")


if __name__ == "__main__":
    main()
