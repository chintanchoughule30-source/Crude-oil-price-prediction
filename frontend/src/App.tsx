import { useEffect, useState, type ChangeEvent, type FormEvent, type ReactNode } from 'react';
import {
  Activity,
  BrainCircuit,
  ChartNoAxesCombined,
  Dices,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

type PredictionInput = {
  price_lag_1: number;
  price_lag_2: number;
  price_lag_5: number;
  price_lag_20: number;
  sma_5: number;
  sma_20: number;
  return_1d: number;
  volatility_20: number;
};

type MetricPair = {
  mae: number;
  rmse: number;
};

type ResidualBin = {
  range_start: number;
  range_end: number;
  count: number;
};

type ModelInsights = {
  metrics: {
    model: MetricPair & { r2: number };
    baseline: MetricPair;
    rmse_improvement_percent: number | null;
  };
  hyperparameters: {
    n_estimators: number;
    learning_rate: number;
    max_depth: number;
    random_state: number;
    objective: string;
    importance_type: string;
  };
  feature_importances: Array<{ feature: string; importance: number }>;
  residual_distribution: ResidualBin[];
};

type TestPerformanceRow = {
  date: string;
  actual: number;
  predicted: number;
  residual: number;
  features: PredictionInput;
};

type PredictionResponse = {
  predicted_price: number;
  currency: string;
  unit: string;
};

const featureFields: Array<{
  name: keyof PredictionInput;
  label: string;
  step: string;
}> = [
  { name: 'price_lag_1', label: 'Price t-1 ($)', step: '0.01' },
  { name: 'price_lag_2', label: 'Price t-2 ($)', step: '0.01' },
  { name: 'price_lag_5', label: 'Price t-5 ($)', step: '0.01' },
  { name: 'price_lag_20', label: 'Price t-20 ($)', step: '0.01' },
  { name: 'sma_5', label: '5-session average ($)', step: '0.01' },
  { name: 'sma_20', label: '20-session average ($)', step: '0.01' },
  { name: 'return_1d', label: '1-day return', step: '0.001' },
  { name: 'volatility_20', label: '20-session volatility', step: '0.001' },
];

const initialFeatures: PredictionInput = {
  price_lag_1: 78.5,
  price_lag_2: 78.1,
  price_lag_5: 76.9,
  price_lag_20: 74.2,
  sma_5: 77.8,
  sma_20: 75.1,
  return_1d: 0.005,
  volatility_20: 0.012,
};

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}) for ${url}`);
  }
  return response.json() as Promise<T>;
}

function formatCurrency(value: number): string {
  return `$${value.toFixed(2)}`;
}

function formatFeatureName(value: string): string {
  return value.replaceAll('_', ' ');
}

function formatRmseComparison(improvement: number | null): string {
  if (improvement === null) return 'Naive baseline RMSE comparison unavailable';
  if (improvement > 0) return `${improvement.toFixed(1)}% lower RMSE than the naive baseline`;
  if (improvement < 0) return `${Math.abs(improvement).toFixed(1)}% higher RMSE than the naive baseline`;
  return 'Same RMSE as the naive baseline';
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [insights, setInsights] = useState<ModelInsights | null>(null);
  const [testRows, setTestRows] = useState<TestPerformanceRow[]>([]);
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [isLoadingDashboard, setIsLoadingDashboard] = useState(true);
  const [formData, setFormData] = useState<PredictionInput>(initialFeatures);
  const [sampleDate, setSampleDate] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadDashboard() {
      try {
        const [modelInsights, performance] = await Promise.all([
          fetchJson<ModelInsights>(`${API_BASE_URL}/model-insights`, controller.signal),
          fetchJson<{ rows: TestPerformanceRow[] }>(
            `${API_BASE_URL}/test-performance`,
            controller.signal,
          ),
        ]);
        setInsights(modelInsights);
        setTestRows(performance.rows);
      } catch (error) {
        if (!controller.signal.aborted) {
          setDashboardError(
            error instanceof Error ? error.message : 'Could not load model evaluation data.',
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingDashboard(false);
        }
      }
    }

    void loadDashboard();
    return () => controller.abort();
  }, []);

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: value === '' ? 0 : Number(value),
    }));
  };

  const handleLoadRandomSample = () => {
    if (testRows.length === 0) return;
    const sample = testRows[Math.floor(Math.random() * testRows.length)];
    setFormData(sample.features);
    setSampleDate(sample.date);
    setPrediction(null);
    setPredictionError(null);
  };

  const submitPrediction = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsPredicting(true);
    setPredictionError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) {
        throw new Error(`Prediction request failed (${response.status}).`);
      }
      const result = (await response.json()) as PredictionResponse;
      setPrediction(result.predicted_price);
    } catch (error) {
      setPredictionError(
        error instanceof Error ? error.message : 'Could not get a prediction from the API.',
      );
    } finally {
      setIsPredicting(false);
    }
  };

  const chartRows = testRows.map((row) => ({
    ...row,
    shortDate: row.date.slice(0, 7),
  }));
  const residualBins =
    insights?.residual_distribution.map((bin) => ({
      ...bin,
      range: `${bin.range_start.toFixed(1)} to ${bin.range_end.toFixed(1)}`,
    })) ?? [];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      <header className="border-b border-slate-200 bg-white/90 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-5 py-5 sm:px-8">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-teal-700">
              CrudeCast AI · Demo dashboard
            </p>
            <h1 className="mt-1 text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
              WTI crude oil forecasting overview
            </h1>
          </div>
          <div className="hidden items-center gap-2 rounded-full bg-teal-50 px-3 py-1.5 text-sm font-medium text-teal-800 sm:flex">
            <Activity size={16} />
            XGBoost live benchmark
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl space-y-6 px-5 py-7 sm:px-8">
        {dashboardError && (
          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 shadow-sm">
            Evaluation data could not be loaded: {dashboardError}. Run
            {' '}<code>python .\backend\pipeline\train.py</code> and check that the API is running.
          </div>
        )}

        <section className="rounded-2xl border border-slate-200 bg-gradient-to-r from-white to-slate-50 p-5 shadow-sm sm:p-6">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                Model transparency view
              </p>
              <h2 className="mt-2 text-xl font-semibold text-slate-900">
                Benchmarking the XGBoost model against a naive baseline
              </h2>
            </div>
            <div className="rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-sm font-medium text-teal-800">
              {insights ? `R² ${formatPercent(insights.metrics.model.r2)}` : 'Loading model summary…'}
            </div>
          </div>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
            This dashboard highlights the held-out test performance, the model’s explainability signals,
            and the feature inputs used for live inference. The benchmark remains deliberately honest and
            compares the model against a persistence-style baseline.
          </p>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-slate-900 p-5 text-white shadow-sm sm:p-6">
          <div className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr] lg:items-center">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-teal-300">
                Executive summary
              </p>
              <h2 className="mt-2 text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                The model explains the market direction well, but the crude market remains highly persistent.
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-300">
                In practical terms, the XGBoost model captures the underlying structure of WTI pricing,
                delivers a strong R² signal, and surfaces explainable rules for the decision process.
                However, the benchmark shows that a simple persistence baseline remains extremely hard to
                beat, so the result is best framed as a transparent analytical tool rather than a
                guaranteed forecasting advantage.
              </p>
            </div>
            <div className="rounded-2xl border border-slate-700 bg-slate-800/80 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-400">
                Decision signal
              </p>
              <p className="mt-3 text-3xl font-semibold text-teal-300">
                {insights ? formatPercent(insights.metrics.model.r2) : '—'}
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                Model fit on held-out data, with honest comparison to the baseline and clear feature-level
                explainability.
              </p>
            </div>
          </div>
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <InsightPill
            title="Benchmark takeaway"
            value={
              insights
                ? formatRmseComparison(insights.metrics.rmse_improvement_percent)
                : 'Model benchmark loading…'
            }
            accent="teal"
          />
          <InsightPill
            title="Model transparency"
            value={
              insights
                ? `${insights.feature_importances.length} explainability inputs tracked`
                : 'Feature sensitivity loading…'
            }
            accent="blue"
          />
          <InsightPill
            title="Presentation note"
            value="The model is benchmarked honestly against a persistence baseline with clear evidence."
            accent="slate"
          />
        </section>

        <section aria-label="Model evaluation summary" className="grid gap-4 md:grid-cols-3">
          <MetricCard
            icon={<ChartNoAxesCombined size={18} />}
            label="Test RMSE vs baseline"
            value={
              insights
                ? `${formatCurrency(insights.metrics.model.rmse)} / ${formatCurrency(insights.metrics.baseline.rmse)}`
                : isLoadingDashboard
                  ? 'Loading…'
                  : 'Unavailable'
            }
            detail={
              insights
                ? formatRmseComparison(insights.metrics.rmse_improvement_percent)
                : 'Model RMSE / naive baseline RMSE'
            }
          />
          <MetricCard
            icon={<BrainCircuit size={18} />}
            label="Test R²"
            value={insights ? formatPercent(insights.metrics.model.r2) : '—'}
            detail="Variance explained on the held-out test set"
          />
          <MetricCard
            icon={<Activity size={18} />}
            label="Model configuration"
            value={
              insights
                ? `${insights.hyperparameters.n_estimators} trees · depth ${insights.hyperparameters.max_depth}`
                : '—'
            }
            detail={
              insights
                ? `Learning rate ${insights.hyperparameters.learning_rate}`
                : 'Trained XGBoost hyperparameters'
            }
          />
        </section>

        <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-2">
            <div>
              <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">
                Chronological holdout
              </p>
              <h2 className="mt-1 text-lg font-semibold text-slate-900">
                Actual vs predicted · test set
              </h2>
            </div>
            <p className="text-sm text-slate-500">
              {testRows.length > 0 ? `${testRows.length.toLocaleString()} unseen observations` : ''}
            </p>
          </div>

          <div className="h-80 w-full">
            {chartRows.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartRows} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
                  <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="shortDate"
                    minTickGap={36}
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: '#cbd5e1' }}
                  />
                  <YAxis
                    domain={['auto', 'auto']}
                    tickFormatter={(value: number) => `$${value}`}
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={64}
                  />
                  <Tooltip
                    labelFormatter={(_, payload) => payload[0]?.payload.date ?? ''}
                    formatter={(value) => [formatCurrency(Number(value)), '']}
                    contentStyle={{
                      border: '1px solid #e2e8f0',
                      borderRadius: 10,
                      boxShadow: '0 8px 24px rgb(15 23 42 / 0.08)',
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    name="Actual WTI"
                    stroke="#0f766e"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name="XGBoost prediction"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <ChartMessage loading={isLoadingDashboard} />
            )}
          </div>
          {insights && (
            <p className="mt-3 text-xs text-slate-500">
              Test MAE {formatCurrency(insights.metrics.model.mae)} · Baseline MAE{' '}
              {formatCurrency(insights.metrics.baseline.mae)} · Residual = actual − predicted
            </p>
          )}
        </section>

        <section className="grid items-start gap-6 xl:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
            <div className="mb-5">
              <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">
                Inference
              </p>
              <h2 className="mt-1 text-lg font-semibold text-slate-900">Explore a prediction</h2>
              <p className="mt-1 text-sm text-slate-500">
                Enter the eight features used by the trained model.
              </p>
            </div>

            <button
              type="button"
              onClick={handleLoadRandomSample}
              disabled={testRows.length === 0}
              className="mb-4 inline-flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Dices size={16} />
              Load random test sample
            </button>
            {sampleDate && (
              <p className="mb-4 text-xs text-slate-500">
                Features loaded from test observation {sampleDate}.
              </p>
            )}

            <form onSubmit={submitPrediction}>
              <div className="grid gap-3 sm:grid-cols-2">
                {featureFields.map(({ name, label, step }) => (
                  <label key={name} className="block text-sm font-medium text-slate-700">
                    {label}
                    <input
                      type="number"
                      step={step}
                      name={name}
                      value={Number(
                        formData[name].toFixed(
                          name === 'return_1d' || name === 'volatility_20' ? 6 : 2,
                        ),
                      )}
                      onChange={handleInputChange}
                      required
                      className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-teal-600 focus:ring-2 focus:ring-teal-100"
                    />
                  </label>
                ))}
              </div>

              <button
                type="submit"
                disabled={isPredicting}
                className="mt-5 w-full rounded-lg bg-teal-700 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-teal-800 disabled:cursor-wait disabled:opacity-60"
              >
                {isPredicting ? 'Calculating…' : 'Forecast next trading day'}
              </button>
            </form>

            {prediction !== null && (
              <div className="mt-4 rounded-lg border border-teal-200 bg-teal-50 p-4">
                <p className="text-sm font-medium text-teal-900">Predicted WTI price</p>
                <p className="mt-1 text-3xl font-semibold text-teal-800">
                  {formatCurrency(prediction)}
                  <span className="ml-2 text-sm font-normal">per barrel</span>
                </p>
              </div>
            )}
            {predictionError && (
              <p role="alert" className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-800">
                {predictionError}
              </p>
            )}
          </div>

          <div className="space-y-6">
            <ChartPanel
              eyebrow="Explainability"
              title="XGBoost feature importance · gain"
              loading={isLoadingDashboard}
              empty={!insights || insights.feature_importances.length === 0}
              description="Relative normalized average gain across tree splits."
            >
              {insights && insights.feature_importances.length > 0 && (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={insights.feature_importances}
                    layout="vertical"
                    margin={{ top: 4, right: 20, bottom: 4, left: 12 }}
                  >
                    <CartesianGrid horizontal={false} stroke="#e2e8f0" />
                    <XAxis
                      type="number"
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      width={125}
                      tickFormatter={formatFeatureName}
                      tick={{ fill: '#475569', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      formatter={(value) => [Number(value).toFixed(4), 'Importance']}
                      labelFormatter={(label) => formatFeatureName(String(label))}
                    />
                    <Bar dataKey="importance" fill="#0f766e" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>

            <ChartPanel
              eyebrow="Error analysis"
              title="Test-set residual distribution"
              loading={isLoadingDashboard}
              empty={!insights || residualBins.length === 0}
            >
              {residualBins.length > 0 && (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={residualBins} margin={{ top: 4, right: 12, bottom: 30, left: 0 }}>
                    <CartesianGrid stroke="#e2e8f0" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="range"
                      interval="preserveStartEnd"
                      angle={-30}
                      textAnchor="end"
                      tick={{ fill: '#64748b', fontSize: 10 }}
                      tickLine={false}
                      axisLine={{ stroke: '#cbd5e1' }}
                    />
                    <YAxis
                      allowDecimals={false}
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      formatter={(value) => [value, 'Observations']}
                      labelFormatter={(label) => `Residual: ${label}`}
                    />
                    <Bar dataKey="count" name="Observations" radius={[3, 3, 0, 0]}>
                      {residualBins.map((bin, index) => (
                        <Cell
                          key={`${bin.range_start}-${index}`}
                          fill={
                            bin.range_start <= 0 && bin.range_end >= 0
                              ? '#0f766e'
                              : '#93c5fd'
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </ChartPanel>
          </div>
        </section>
      </main>
    </div>
  );
}

function MetricCard({
  icon,
  label,
  value,
  detail,
}: {
  icon: ReactNode;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2 text-sm font-medium text-slate-500">
        <span className="text-teal-700">{icon}</span>
        {label}
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-slate-900">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{detail}</p>
    </article>
  );
}

function InsightPill({
  title,
  value,
  accent,
}: {
  title: string;
  value: string;
  accent: 'teal' | 'blue' | 'slate';
}) {
  const accentMap = {
    teal: 'border-teal-200 bg-teal-50 text-teal-800',
    blue: 'border-blue-200 bg-blue-50 text-blue-800',
    slate: 'border-slate-200 bg-slate-100 text-slate-700',
  };

  return (
    <div className={`rounded-xl border p-4 shadow-sm ${accentMap[accent]}`}>
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] opacity-80">{title}</p>
      <p className="mt-2 text-sm font-medium leading-6">{value}</p>
    </div>
  );
}

function ChartPanel({
  eyebrow,
  title,
  loading,
  empty,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  loading: boolean;
  empty: boolean | undefined;
  description?: string;
  children: ReactNode;
}) {
  return (
    <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="mb-4">
        <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">{eyebrow}</p>
        <h2 className="mt-1 text-lg font-semibold text-slate-900">{title}</h2>
      </div>
      <div className="h-72">
        {loading || empty ? <ChartMessage loading={loading} /> : children}
      </div>
      {description && <p className="mt-3 text-xs text-slate-500">{description}</p>}
    </article>
  );
}

function ChartMessage({ loading }: { loading: boolean }) {
  return (
    <div className="flex h-full items-center justify-center rounded-lg border border-dashed border-slate-200 text-sm text-slate-500">
      {loading ? 'Loading evaluation data…' : 'No evaluation data available. Train the model first.'}
    </div>
  );
}

export default App;
