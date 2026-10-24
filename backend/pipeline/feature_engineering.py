from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def engineer_features(
    input_path=PROJECT_ROOT / "data" / "raw" / "wti_prices.csv",
    output_path=PROJECT_ROOT / "data" / "processed" / "features.csv",
):
    """Create the historical price features used by the model."""
    df = pd.read_csv(input_path)

    # FRED names the date column "observation_date"; the pipeline uses "DATE".
    if "DATE" not in df.columns and "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "DATE"})
    if "DATE" not in df.columns:
        raise ValueError("Input data must contain a DATE or observation_date column.")

    # Sort chronologically before calculating any features.
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").set_index("DATE")
    df = df.rename(columns={"DCOILWTICO": "price"})
    price = df["price"]

    # Lags use earlier prices; moving averages summarize recent prices.
    df["price_lag_1"] = price.shift(1)
    df["price_lag_2"] = price.shift(2)
    df["price_lag_5"] = price.shift(5)
    df["price_lag_20"] = price.shift(20)
    df["sma_5"] = price.rolling(window=5).mean()
    df["sma_20"] = price.rolling(window=20).mean()

    # Returns measure daily price changes; volatility measures their variation.
    df["return_1d"] = price.pct_change()
    df["volatility_20"] = df["return_1d"].rolling(window=20).std()

    # The target is the next available price, which the model learns to predict.
    df["target"] = price.shift(-1)

    # Early rows lack history for the rolling features; the last row lacks a target.
    df.dropna(inplace=True)

    # Create the destination folder if this is the first pipeline run.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    print(f"Engineered dataset saved to {output_path}. Shape: {df.shape}")


if __name__ == "__main__":
    engineer_features()