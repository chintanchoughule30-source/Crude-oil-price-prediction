import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if __package__:
    from .features import MODEL_FEATURES
else:
    from features import MODEL_FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES = MODEL_FEATURES
HYPERPARAMETERS = {
    "n_estimators": 150,
    "learning_rate": 0.05,
    "max_depth": 5,
    "random_state": 42,
    "objective": "reg:squarederror",
    "importance_type": "gain",
}


def train_and_evaluate(
    data_path=PROJECT_ROOT / "data" / "processed" / "features.csv",
    model_dir=PROJECT_ROOT / "artifacts",
):
    """Train the model, measure its accuracy, and save its results."""
    print("Loading engineered features...")
    df = pd.read_csv(data_path, index_col="DATE", parse_dates=True)

    # Keep earlier dates first so future data never leaks into training.
    train_end = int(len(df) * 0.70)
    test_start = int(len(df) * 0.85)

    # iloc selects rows by position; the end of each slice is not included.
    train_data = df.iloc[:train_end]
    validation_data = df.iloc[train_end:test_start]
    test_data = df.iloc[test_start:]

    X_train, y_train = train_data[FEATURES], train_data["target"]
    X_val, y_val = validation_data[FEATURES], validation_data["target"]
    X_test, y_test = test_data[FEATURES], test_data["target"]

    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(**HYPERPARAMETERS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Compare predictions with the known prices in the held-out test period.
    predictions = model.predict(X_test)
    residuals = y_test.to_numpy() - predictions
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Baseline prediction: tomorrow's price is the latest known price.
    baseline_predictions = test_data["price"]
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_predictions))
    rmse_improvement = (
        (baseline_rmse - rmse) / baseline_rmse * 100 if baseline_rmse else None
    )

    print("\n--- Final Test Evaluation ---")
    print(f"XGBoost MAE:       ${mae:.2f}")
    print(f"XGBoost RMSE:      ${rmse:.2f}")
    print(f"XGBoost R^2 Score: {r2:.3f}")
    print(f"Baseline MAE:      ${baseline_mae:.2f}")
    print(f"Baseline RMSE:     ${baseline_rmse:.2f}")
    mae_comparison = "lower" if mae < baseline_mae else "higher"
    rmse_comparison = "lower" if rmse < baseline_rmse else "higher"
    print(f"XGBoost MAE is {mae_comparison} than the naive baseline.")
    print(f"XGBoost RMSE is {rmse_comparison} than the naive baseline.")

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

    # Divide test errors into 12 ranges for the residual histogram.
    histogram_counts, histogram_edges = np.histogram(residuals, bins=12)

    # Show the most useful features first.
    feature_importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    # Save each test date, prediction, and its input values for the API and chart.
    test_performance = []
    for (date, feature_values), actual, predicted, residual in zip(
        X_test.iterrows(),
        y_test,
        predictions,
        residuals,
    ):
        test_performance.append({
            "date": date.strftime("%Y-%m-%d"),
            "actual": float(actual),
            "predicted": float(predicted),
            "residual": float(residual),
            "features": {
                feature: float(value)
                for feature, value in zip(FEATURES, feature_values)
            },
        })

    evaluation_artifacts = {
        "metrics": {
            "model": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
            "baseline": {
                "mae": float(baseline_mae),
                "rmse": float(baseline_rmse),
            },
            "rmse_improvement_percent": (
                float(rmse_improvement) if rmse_improvement is not None else None
            ),
        },
        "hyperparameters": HYPERPARAMETERS,
        # Convert NumPy numbers to regular Python numbers so JSON can save them.
        "feature_importances": [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in feature_importances
        ],
        "residual_distribution": [
            {
                "range_start": float(histogram_edges[index]),
                "range_end": float(histogram_edges[index + 1]),
                "count": int(count),
            }
            for index, count in enumerate(histogram_counts)
        ],
        "test_performance": test_performance,
    }

    artifacts_path = model_dir / "evaluation_artifacts.json"
    with artifacts_path.open("w", encoding="utf-8") as artifacts_file:
        json.dump(evaluation_artifacts, artifacts_file, indent=2)
    print(f"Evaluation artifacts saved to {artifacts_path}")


if __name__ == "__main__":
    train_and_evaluate()
