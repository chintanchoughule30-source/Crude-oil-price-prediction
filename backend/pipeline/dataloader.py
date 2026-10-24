from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def download_fred_data(series_id="DCOILWTICO", output_dir=PROJECT_ROOT / "data" / "raw"):
    """Download a FRED price series and save it as a CSV file."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {series_id} from FRED...")
    # FRED uses "." for missing prices, such as on non-trading days.
    df = pd.read_csv(url, na_values=".")
    df.dropna(inplace=True)

    output_path = output_dir / "wti_prices.csv"
    df.to_csv(output_path, index=False)
    print(f"Raw data successfully saved to {output_path}")


if __name__ == "__main__":
    download_fred_data()