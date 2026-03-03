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
    domestic_multiplier?: Metric & { worst_examples?: Array<{ title: string; abs_pct_error: number }> };
    intl_dom_ratio?: Metric;
    combined_totals?: {
      domestic?: Metric;
      international?: Metric;
      worldwide?: Metric;
    };
  };
  backtest?: {
    domestic_multiplier?: { metrics: Metric };
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

function StatCard({ title, value, subtitle }: { title: string; value: string; subtitle?: string }) {
  return (
    <div className="card stat-card">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      {subtitle ? <div className="stat-subtitle">{subtitle}</div> : null}
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
    <div className="dashboard-grid">
      <StatCard
        title="Rows (Filtered)"
        value={`${report.rows.training_examples_after_filters ?? 0}`}
        subtitle={`Raw: ${report.rows.training_examples_before_filters ?? 0}`}
      />
      <StatCard
        title="Domestic MAE"
        value={fmtMoney(report.models.combined_totals?.domestic?.mae ?? 0)}
        subtitle={`MAPE ${(report.models.combined_totals?.domestic?.mape_pct ?? 0).toFixed(1)}%`}
      />
      <StatCard
        title="Worldwide MAE"
        value={fmtMoney(report.models.combined_totals?.worldwide?.mae ?? 0)}
        subtitle={`MAPE ${(report.models.combined_totals?.worldwide?.mape_pct ?? 0).toFixed(1)}%`}
      />
      <StatCard
        title="Backtest Domestic MAE"
        value={(report.backtest?.domestic_multiplier?.metrics?.mae ?? 0).toFixed(3)}
        subtitle={`MAPE ${(report.backtest?.domestic_multiplier?.metrics?.mape_pct ?? 0).toFixed(1)}%`}
      />

      <div className="card chart-card">
        <h3>Total MAE by Target</h3>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={maeBars}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="metric" />
            <YAxis />
            <Tooltip formatter={(v: number) => fmtMoney(v)} />
            <Legend />
            <Bar dataKey="mae" fill="#1f6feb" name="MAE" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card chart-card">
        <h3>Monthly Domestic MAE</h3>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={monthly}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" />
            <YAxis />
            <Tooltip formatter={(v: number) => fmtMoney(v)} />
            <Legend />
            <Line type="monotone" dataKey="mae" stroke="#0f766e" strokeWidth={3} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card table-card">
        <h3>Worst Domestic Multiplier Errors</h3>
        <table>
          <thead>
            <tr>
              <th>Title</th>
              <th>Error %</th>
            </tr>
          </thead>
          <tbody>
            {worst.map((row) => (
              <tr key={row.title}>
                <td>{row.title}</td>
                <td>{row.abs_pct_error.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
