# CrudeCast AI: Easy Project Explanation

## 1. What is this project?

CrudeCast AI is a small website that uses past oil prices to estimate the next
available WTI oil price.

It has three main parts:

1. Python prepares the price data and trains a model.
2. FastAPI provides the model through a web API.
3. A React website shows charts and lets you try a prediction.

The predicted price is in **US dollars per barrel**. This is an educational
project, not financial advice. It cannot guarantee future prices.

## 2. What are FRED and WTI?

- **FRED** is a public database run by the Federal Reserve Bank of St. Louis.
  This project downloads its oil-price data.
- **WTI** means West Texas Intermediate. It is a common US crude-oil
  benchmark.
- The project uses FRED series **`DCOILWTICO`**, which records WTI spot prices.
  A spot price is the price of oil now, not the price of a future contract.

The data gives a date and a price in dollars per barrel. The data is mainly
for trading days, so the next row may be Monday after Friday rather than the
next calendar day.

This dataset is public, has many years of prices, and is useful for a student
forecasting project. But it contains prices only. It does not include news,
weather, oil supply, demand, or world events that can also affect prices.

In the saved data snapshot, there are 9,517 raw rows and 9,496 processed rows.
These counts and dates can change when the data is downloaded again.

## 3. How does the project work?

Run these three Python files in order:

1. **`backend/pipeline/dataloader.py`** downloads the FRED data to
   `data/raw/wti_prices.csv`.
2. **`backend/pipeline/feature_engineering.py`** cleans the data and creates
   the model inputs. It saves them in `data/processed/features.csv`.
3. **`backend/pipeline/train.py`** splits the data, trains and checks the
   model, and saves the results in `artifacts/`.

The first step needs an internet connection.

## 4. What information does the model use?

The model uses eight features. A **feature** is one piece of information
given to the model to help it make a prediction.

| Feature | Simple meaning |
| --- | --- |
| `price_lag_1` | Price at the previous available observation |
| `price_lag_2` | Price two observations ago |
| `price_lag_5` | Price five observations ago |
| `price_lag_20` | Price twenty observations ago |
| `sma_5` | Average price over the latest five observations |
| `sma_20` | Average price over the latest twenty observations |
| `return_1d` | How much the price changed from the previous observation |
| `volatility_20` | How much recent price changes have varied over twenty observations |

The model uses these values to predict the **next available price**. The
first rows are removed because they do not have enough earlier prices to
calculate every feature. The last row is removed because its next price is
not yet in the data.

The project chooses these features by writing formulas in the code. It does
not use a separate automatic feature-selection method. The importance chart
shows which features the trained model used most, but that does not prove
that a feature causes prices to change.

## 5. How is the data split?

The rows stay in date order. The project does not shuffle them:

| Part | Share | What it is for |
| --- | ---: | --- |
| Training | 70% | The model learns from these rows |
| Validation | 15% | Used as an extra check while fitting |
| Testing | 15% | Used to measure results on later dates |

The saved data has 9,496 processed rows: 6,647 for training, 1,424 for
validation, and 1,425 for testing. The test period starts on 2021-01-22 in
this snapshot.

Keeping later dates for testing is important. It checks whether a model
trained on the past can work on newer data. Mixing dates randomly could let
the model learn from dates that come after the dates being tested.

## 6. What is XGBoost?

This project uses **XGBoost regression**.

- **Regression** means predicting a number, such as an oil price.
- XGBoost builds many small decision trees. Each new tree tries to improve
  the earlier trees' results.
- A decision tree makes choices using rules, such as whether a feature is
  above or below a value.

XGBoost is useful for this project because it works with number-based tables
and can learn patterns that are not simple straight lines. It does not
automatically know which date comes first; the past-price features provide
that information.

The project uses these settings, called **hyperparameters**:

| Setting | Value | Simple meaning |
| --- | ---: | --- |
| `n_estimators` | 150 | Number of trees |
| `learning_rate` | 0.05 | How much each new tree changes the result |
| `max_depth` | 5 | How many levels deep a tree can grow |
| `random_state` | 42 | Helps make runs repeatable |
| `objective` | `reg:squarederror` | Trains the model to reduce squared errors |
| `importance_type` | `gain` | How the feature-importance values are measured |

These settings are fixed in the code. The project does not test many settings
to find the best ones.

## 7. How well did the model do?

For price predictions, there is no simple classification “accuracy”
percentage. A prediction can be close or far from the real price. This
project reports three measures:

| Measure | XGBoost result | What it means |
| --- | ---: | --- |
| MAE | $1.65 per barrel | Average size of the prediction error |
| RMSE | $2.38 per barrel | Error measure that penalizes big mistakes more |
| R² | 0.969 | How much of the test price variation the model explains compared with using the average |

These are results from the currently saved test data. Retraining can change
them.

**Important:** R² of 0.969 does not mean that 96.9% of predictions are
correct.

### What is the baseline?

A baseline is a simple result used for comparison. This project uses a
**persistence baseline**: it predicts that the next price will be the same
as the latest known price.

| Measure | XGBoost | Simple baseline |
| --- | ---: | ---: |
| MAE | $1.65 | $1.52 |
| RMSE | $2.38 | $2.23 |

Lower errors are better. In this saved result, the simple baseline does
better than XGBoost. XGBoost's RMSE is about 6.4% higher. So the project
should not claim that XGBoost beats the baseline, even though its R² is high.

### Is the model overfit or underfit?

We cannot tell for sure from the saved results. The project does not show
training and validation scores next to the test score.

- **Overfit** means the model learned the training data very well but does
  worse on new data.
- **Underfit** means the model did not learn the patterns well enough,
  including in its training data.

To check this, calculate the same scores for training, validation, and test
data, then compare them. The final test set should not be used to keep
changing the model.

## 8. What is on the website?

The dashboard has:

- A heading and short model summary.
- Cards showing the test scores and model settings.
- Three charts to understand the predictions and errors.
- A form to try a prediction using the eight features.
- A button to fill the form with a random example from the saved test data.

The random example is an old test observation, not a new oil price. The
frontend gets chart data and results from the backend API.

## 9. What do the charts show?

### Actual vs predicted price

This line chart compares real prices with the model's predictions over the
test dates.

- One line shows the actual WTI price.
- The other line shows the XGBoost prediction.
- Lines that are closer together mean smaller errors on those dates.

This is historical test data, not a live forecast. The chart does not show
how uncertain a prediction is.

### Feature importance

This bar chart ranks the features by how much they helped the model's tree
rules, using a measure called **gain**.

In the saved results, the most important features are:

1. `sma_5`: about 71.6%.
2. `price_lag_1`: about 27.2%.
3. `price_lag_2`: about 1.0%.

These numbers describe how the model used the features. They do not show
cause and effect.

### Residual distribution

A **residual** is the real price minus the predicted price:

`residual = actual price - predicted price`

The chart groups the test errors into ranges and counts how many errors fall
in each range.

- Near zero means the prediction was close.
- Positive means the model predicted too low.
- Negative means the model predicted too high.

The line chart shows when errors happened. The residual chart shows the
overall sizes and directions of errors.

## 10. How does the prediction form work?

Enter the eight features, or select **Load random test sample** to use an
example from the test data. Click **Forecast next trading day** to send the
values to the backend. The backend passes them to the saved model and
returns a price in US dollars per barrel.

The result depends on the values entered. It should not be treated as a
guaranteed real-world forecast.

## 11. What do the main code files do?

| File | What it does |
| --- | --- |
| `backend/pipeline/dataloader.py` | Downloads and saves FRED prices |
| `backend/pipeline/feature_engineering.py` | Creates the eight features and next-price target |
| `backend/pipeline/train.py` | Splits data, trains the model, measures results, and saves files |
| `backend/main.py` | Runs the API and makes predictions |
| `frontend/src/App.tsx` | Shows the dashboard, charts, and prediction form |

The backend API has these routes:

| Route | Purpose |
| --- | --- |
| `GET /health` | Checks whether the API and model are available |
| `GET /model-insights` | Sends the scores, settings, feature importance, and residual data |
| `GET /test-performance` | Sends the test predictions used in the chart |
| `POST /predict` | Returns a prediction for submitted features |

## 12. How is XGBoost different from other models?

The project only trains XGBoost. Other models below are examples for
comparison; this project does not include results for them.

| Model | Simple description |
| --- | --- |
| Persistence baseline | Repeats the latest known price; simple but strong for prices that change slowly |
| Linear regression | Fits a straight-line relationship; easy to explain |
| Random forest | Combines many decision trees trained in parallel |
| XGBoost | Builds trees one after another, with each trying to improve the last |
| ARIMA | A statistical method designed for time-ordered data |
| LSTM | A neural network that can learn from sequences of past values |

To compare models fairly, use the same dates for training and testing, and
compare the same measures such as MAE and RMSE.

## 13. Limitations and possible improvements

- Add other useful information, such as oil supply or demand, if reliable
  data is available.
- Test the model on several later time periods.
- Show scores for training, validation, and testing.
- Try other model settings and compare models using validation data.
- Keep the final test data separate until the final evaluation.
- Show uncertainty as well as one predicted price.

## 14. Run the project on Windows

Open a PowerShell terminal in the project folder. Install dependencies if
they are not already installed:

```powershell
cd "C:\Users\CHINTAN\Desktop\crude oilttttt"
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Prepare the data and model:

```powershell
python .\backend\pipeline\dataloader.py
python .\backend\pipeline\feature_engineering.py
python .\backend\pipeline\train.py
```

Start the backend in the first terminal:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the frontend in a second terminal:

```powershell
cd .\frontend
npm install
npm run dev
```

Open the local web address printed by Vite, usually
`http://localhost:5173`. The API documentation is at
`http://127.0.0.1:8000/docs`.

## 15. Short viva questions and answers

### What does your project predict?

It estimates the next available WTI spot price using recent historical
prices.

### Why did you choose FRED and WTI?

FRED provides public historical data, and WTI is a well-known oil-price
benchmark with many observations.

### Why use XGBoost?

It can learn useful patterns from number-based data and can model
relationships that are not straight lines.

### Why split data by date?

In real forecasting, we use the past to predict the future. Keeping later
dates for testing better represents that situation.

### Is the model better than the simple baseline?

Not in the saved results. The baseline has lower MAE and RMSE, so it makes
smaller average errors in this test.

### What is the main limitation?

The model only uses past WTI prices. It does not know about events or other
market information that can change oil prices.

### What would you improve?

I would test more models, compare their results on the same future dates,
and include more reliable oil-market information.

## 16. Main generated files

- `data/raw/wti_prices.csv`: downloaded prices.
- `data/processed/features.csv`: prepared features and target.
- `artifacts/xgboost_model.joblib`: trained model.
- `artifacts/evaluation_artifacts.json`: scores and data used by the charts.
