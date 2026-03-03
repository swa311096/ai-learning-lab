import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

type Metric = {
  count: number;
  mae: number;
  rmse: number;
  mape_pct: number;
};

type Report = {
  rows: Record<string, number>;
  models: {
    domestic_multiplier?: Metric & {
      worst_examples?: Array<{
        title: string;
        abs_pct_error: number;
        actual?: number;
        predicted?: number;
        release_date?: string;
        opening_weekend_usd?: number;
        domestic_total_usd?: number;
        day3_total_usd?: number;
        day7_total_usd?: number;
      }>;
      multiplier_formula?: string;
    };
    intl_dom_ratio?: Metric;
    combined_totals?: {
      domestic?: Metric;
      international?: Metric;
      worldwide?: Metric;
    };
  };
  backtest?: {
    domestic_multiplier?: { metrics: Metric; test_count?: number };
    intl_dom_ratio?: { metrics: Metric };
  };
  charts?: {
    domestic_total_monthly_mae?: Array<{ month: number; count: number; mae: number }>;
  };
};

function fmtMoney(value: number): string {
  if (!Number.isFinite(value)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function StatCard({
  title,
  value,
  subtitle,
  description,
}: {
  title: string;
  value: string;
  subtitle?: string;
  description?: string;
}) {
  return (
    <div className="card stat-card">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      {subtitle ? <div className="stat-subtitle">{subtitle}</div> : null}
      {description ? <div className="stat-desc">{description}</div> : null}
    </div>
  );
}

export function MetricsDashboard({ report }: { report: Report }) {
  const maeBars = useMemo(() => {
    const domestic = report.models.combined_totals?.domestic?.mae ?? 0;
    const intl = report.models.combined_totals?.international?.mae ?? 0;
    const worldwide = report.models.combined_totals?.worldwide?.mae ?? 0;
    return [
      { metric: "Domestic", mae: domestic },
      { metric: "International", mae: intl },
      { metric: "Worldwide", mae: worldwide },
    ];
  }, [report]);

  const monthly = report.charts?.domestic_total_monthly_mae ?? [];
  const worst = report.models.domestic_multiplier?.worst_examples ?? [];

  return (
    <>
      <h2 className="section-header">1. The data</h2>
      <p className="section-desc">
        How many movies we trained on. Some were filtered out (e.g. tiny openings, outliers).
      </p>
      <div className="dashboard-grid">
        <StatCard
          title="Movies used for training"
          value={`${report.rows.training_examples_after_filters ?? 0}`}
          subtitle={`Before filtering: ${report.rows.training_examples_before_filters ?? 0}`}
          description="Each movie has opening weekend + 7 days of data."
        />
      </div>

      <h2 className="section-header">2. In-sample accuracy (how well we fit the data we trained on)</h2>
      <p className="section-desc">
        On average, how far off are our predictions? Lower numbers = better.
      </p>
      <div className="dashboard-grid">
        <StatCard
          title="Domestic box office — average error"
          value={fmtMoney(report.models.combined_totals?.domestic?.mae ?? 0)}
          subtitle={`~${(report.models.combined_totals?.domestic?.mape_pct ?? 0).toFixed(0)}% off on average`}
          description="Predicting US/Canada total from opening weekend."
        />
        <StatCard
          title="Worldwide box office — average error"
          value={fmtMoney(report.models.combined_totals?.worldwide?.mae ?? 0)}
          subtitle={`~${(report.models.combined_totals?.worldwide?.mape_pct ?? 0).toFixed(0)}% off on average`}
          description="Predicting global total (domestic + international)."
        />
      </div>

      <h2 className="section-header">3. Backtest (train on old movies, test on new — more realistic)</h2>
      <p className="section-desc">
        We pretended we only had data up to a certain date, trained on that, then predicted movies that came out later.
      </p>
      <div className="dashboard-grid">
        <StatCard
          title="Domestic multiplier backtest"
          value={(report.backtest?.domestic_multiplier?.metrics?.mae ?? 0).toFixed(2)}
          subtitle={`MAPE ~${(report.backtest?.domestic_multiplier?.metrics?.mape_pct ?? 0).toFixed(0)}% on ${report.backtest?.domestic_multiplier?.test_count ?? 0} held-out movies`}
          description="Multiplier = final domestic ÷ opening weekend. This tests generalization."
        />
      </div>

      <h2 className="section-header">4. Error by target type</h2>
      <p className="section-desc">
        Domestic predictions are usually more accurate than international or worldwide.
      </p>
      <div className="dashboard-grid">
      <div className="card chart-card">
        <h3>Average dollar error by target</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={maeBars}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis tickFormatter={(v) => `$${(v / 1e6).toFixed(0)}M`} />
            <Tooltip formatter={(v: number) => fmtMoney(v)} />
            <Legend />
            <Bar dataKey="mae" fill="#1f6feb" name="MAE" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      </div>

      <h2 className="section-header">5. Does accuracy vary by release month?</h2>
      <p className="section-desc">
        Some months (e.g. holidays) may be easier or harder to predict.
      </p>
      <div className="dashboard-grid">
      <div className="card chart-card">
        <h3>Domestic prediction error by release month</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={monthly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" tickFormatter={(m) => ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m - 1] ?? m} />
            <YAxis tickFormatter={(v) => `$${(v / 1e6).toFixed(0)}M`} />
            <Tooltip formatter={(v: number) => fmtMoney(v)} />
            <Legend />
            <Line type="monotone" dataKey="mae" stroke="#0f766e" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      </div>

      <h2 className="section-header">6. Where do we fail the most?</h2>
      <p className="section-desc">
        Movies where our prediction was furthest from reality. Often niche films, anime, or re-releases.
        {report.models.domestic_multiplier?.multiplier_formula ? (
          <span className="formula-note"> Formula: {report.models.domestic_multiplier.multiplier_formula}</span>
        ) : null}
        <a href="/multiplier_calculation.txt" target="_blank" rel="noopener noreferrer" className="detail-link">
          View full multiplier calculation details (opens in new tab)
        </a>
      </p>
      <div className="dashboard-grid">
      <div className="card table-card">
        <h3>Biggest domestic multiplier misses</h3>
        <table>
          <thead>
            <tr>
              <th>Title</th>
              <th>Weekend</th>
              <th>Total</th>
              <th>Mult. (Total÷Weekend)</th>
              <th>Predicted</th>
              <th>Error</th>
            </tr>
          </thead>
          <tbody>
            {worst.map((row) => (
              <tr key={`${row.title}-${row.release_date ?? ""}`}>
                <td>{row.title}</td>
                <td>{row.opening_weekend_usd != null ? fmtMoney(row.opening_weekend_usd) : "—"}</td>
                <td>{row.domestic_total_usd != null ? fmtMoney(row.domestic_total_usd) : "—"}</td>
                <td>{row.actual != null ? row.actual.toFixed(2) : "—"}</td>
                <td>{row.predicted != null ? row.predicted.toFixed(2) : "—"}</td>
                <td>{row.abs_pct_error.toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
    </>
  );
}
