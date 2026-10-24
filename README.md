# CrudeCast AI

### An end-to-end WTI forecasting and model-evaluation project

CrudeCast AI turns historical West Texas Intermediate (WTI) crude-oil prices
into an interactive forecasting dashboard. It brings together a reproducible
data pipeline, a time-ordered machine-learning evaluation, a typed prediction
API, and a React dashboard.

> **Honest benchmark, not investment advice.** In the currently saved
> evaluation snapshot, XGBoost does not beat the simple persistence baseline.
> The project reports this result rather than presenting a high R² as proof of
> forecasting value. Predictions are experimental and are not financial advice.

**Author:** [Chintan Choughule](https://github.com/chintanchoughule30-source)  
**License:** [MIT](./LICENSE)

## Why this project stands out

- **End to end:** data ingestion, feature engineering, chronological model
  evaluation, API, and an interactive dashboard in one project.
- **Time-aware evaluation:** data is split in date order, preserving a
  genuinely later test period rather than randomly mixing observations.
- **A meaningful baseline:** the model is compared with a persistence
  forecast—the latest known price—and its weaker test result is clearly
  disclosed.
- **Traceable predictions:** test dates, actuals, predictions, residuals, and
  input features power the dashboard's evaluation charts.
- **Practical API safeguards:** typed request and response contracts,
  validation of prediction inputs, explicit CORS configuration, and separate
  liveness and readiness endpoints.
- **Reproducible workflow:** scripts rebuild the dataset features and model
  artifacts; backend contract tests cover the prediction API.

## Project at a glance

| Component | Technology |
| --- | --- |
| Data source | Federal Reserve Bank of St. Louis FRED, `DCOILWTICO` |
| Forecast target | Next available WTI spot-price observation |
| Model | XGBoost `XGBRegressor` |
| Backend | Python, FastAPI, Pydantic |
| Frontend | React, TypeScript, Vite, Recharts |
| Evaluation | Chronological train/validation/test split; MAE, RMSE, R², persistence baseline |
| Price unit | U.S. dollars per barrel |

The model uses eight features calculated from historical WTI prices. It does
not ingest news, inventory, supply-demand indicators, or futures prices.
“Next available observation” means the next price observation in the source
series, not necessarily the next calendar day.

## Evaluation snapshot

The saved evaluation uses a chronological 70% training, 15% validation, and
15% test split. Results below are specific to the artifacts in this repository
and can change when the source data or training configuration changes.

| Test metric | XGBoost | Persistence baseline |
| --- | ---: | ---: |
| MAE | $1.65/barrel | $1.52/barrel |
| RMSE | $2.38/barrel | $2.23/barrel |
| R² | 0.969 | — |

For this snapshot, XGBoost RMSE is approximately **6.4% higher** than the
baseline's. R² describes variance explained relative to a mean prediction; it
does **not** mean that 96.9% of predictions are correct, nor does it establish
that this model is useful for trading. The dashboard presents the baseline
comparison alongside the model metrics.

## How it works

```text
FRED CSV
   │
   ▼
Download and clean ──► Engineer lag, moving-average, return, and volatility features
   │
   ▼
Chronological model training and held-out evaluation
   ├──► XGBoost model + evaluation artifacts ──► FastAPI prediction and insights
   │                                                  │
   └──────────────────────────────────────────────────┴──► React dashboard
```

1. `backend/pipeline/dataloader.py` downloads the FRED series to
   `data/raw/wti_prices.csv`.
2. `backend/pipeline/feature_engineering.py` sorts observations by date,
   calculates the model features and next-observation target, and writes
   `data/processed/features.csv`.
3. `backend/pipeline/train.py` trains XGBoost without shuffling dates,
   evaluates against a persistence baseline, and saves model and evaluation
   artifacts under `artifacts/`.
4. `backend/main.py` serves predictions and evaluation data to the dashboard.

### Model inputs

| Feature | Description |
| --- | --- |
| `price_lag_1`, `price_lag_2`, `price_lag_5`, `price_lag_20` | Prices at earlier available observations |
| `sma_5`, `sma_20` | Rolling mean price over 5 and 20 observations |
| `return_1d` | Percentage change from the previous observation |
| `volatility_20` | Standard deviation of recent returns over 20 observations |

The target is the next available price. The shared feature list in
`backend/pipeline/features.py` keeps the training and API input order aligned.

## Quick start (Windows PowerShell)

### Prerequisites

- Python 3.10 or newer
- Node.js and npm
- An internet connection to download fresh FRED data (optional when using the
  included CSV files)

### 1. Install backend dependencies

Run commands from the project root:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use the environment's Python
executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Rebuild the data and model (optional)

The project includes data and evaluation artifacts for a quick start. To
refresh them, run these commands from the root directory:

```powershell
python .\backend\pipeline\dataloader.py
python .\backend\pipeline\feature_engineering.py
python .\backend\pipeline\train.py
```

The first command downloads data from FRED. Skip it to use the included raw
CSV, or run feature engineering and training to rebuild artifacts from that
snapshot.

### 3. Start the API

In a project-root terminal:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Start the dashboard

In a second terminal:

```powershell
cd .\frontend
npm ci
npm run dev
```

Open the local URL printed by Vite, usually
[http://localhost:5173](http://localhost:5173). The frontend uses
`http://localhost:8000` by default; set `VITE_API_URL` in the frontend
environment to use a different API address.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness and model-loaded status |
| `GET` | `/ready` | Prediction readiness; returns `503` when the model is unavailable |
| `GET` | `/model-insights` | Evaluation metrics, hyperparameters, feature importance, and residual bins |
| `GET` | `/test-performance` | Held-out test rows used by charts and sample input |
| `POST` | `/predict` | Predict a price from the eight model features |

`POST /predict` rejects negative prices, non-finite values, and unexpected
fields. A successful response contains `predicted_price`, `currency` (`USD`),
and `unit` (`Barrel`).

### Configure CORS for a deployment

Local Vite origins are allowed by default. Set `CRUDECAST_CORS_ORIGINS` to
the exact deployed frontend origin(s), separated by commas:

```powershell
$env:CRUDECAST_CORS_ORIGINS = "https://your-dashboard.example"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Tests and quality checks

Run backend contract tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

Run frontend checks from `frontend`:

```powershell
npm run lint
npm run build
```

## Repository layout

```text
backend/
  main.py                    FastAPI application and prediction endpoints
  pipeline/
    dataloader.py            FRED data download
    feature_engineering.py   Historical feature and target creation
    features.py              Canonical model feature order
    train.py                 Chronological training and evaluation
tests/
  test_api.py                Backend input and prediction contract tests
frontend/
  src/                       React dashboard
data/
  raw/                       Source WTI observations
  processed/                 Engineered model dataset
artifacts/
  evaluation_artifacts.json  Metrics, test predictions, and chart data
  xgboost_model.joblib       Locally generated model artifact
LICENSE                      MIT License
PROJECT_EXPLANATION.md       Extended explanation of the model and evaluation
requirements.txt             Python dependencies
```

## Data, artifacts, and responsible use

The data source is the Federal Reserve Bank of St. Louis FRED series
[`DCOILWTICO`](https://fred.stlouisfed.org/series/DCOILWTICO). FRED represents
missing observations with a period; the downloader excludes those rows.
Dataset snapshots and generated model results may become stale, so retrain
before drawing conclusions from them.

This is an educational demonstration, not a production trading system. It
does not provide uncertainty intervals or account for transaction costs,
market structure, macroeconomic events, or operational requirements for
financial services. Do not use its predictions as investment advice.

## License and author

The project code is made available under the [MIT License](./LICENSE).
Copyright © 2026 Chintan Choughule. The license for this code does not
automatically apply to third-party data, libraries, or trademarks; review
their respective terms.

For implementation details, feature definitions, and a deeper discussion of
the model, see [PROJECT_EXPLANATION.md](./PROJECT_EXPLANATION.md).
