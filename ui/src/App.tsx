import { useEffect, useState } from "react";
import { MetricsDashboard } from "./components/MetricsDashboard";
import "./styles.css";

type Report = Parameters<typeof MetricsDashboard>[0]["report"];

export default function App() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/evaluation_report.json")
      .then((res) => {
        if (!res.ok) {
          throw new Error(`failed to load report: ${res.status}`);
        }
        return res.json();
      })
      .then((json) => setReport(json))
      .catch((err) => setError(String(err)));
  }, []);

  return (
    <main className="page">
      <header className="page-header">
        <h1>Box Office Model Dashboard</h1>
        <p>Predicting movie revenue from opening weekend data</p>
      </header>

      <section className="overview-section">
        <h2>How to read this dashboard</h2>
        <ol className="overview-steps">
          <li><strong>What we're predicting:</strong> Given opening weekend numbers, we predict how much a movie will ultimately make domestically and worldwide.</li>
          <li><strong>How we measure accuracy:</strong> <dfn title="Mean Absolute Error — on average, how far off our predictions are in dollars or units">MAE</dfn> = average prediction error. <dfn title="Mean Absolute Percentage Error — on average, how far off we are as a % of the actual value">MAPE</dfn> = average % error. Lower = better.</li>
          <li><strong>In-sample vs backtest:</strong> In-sample = how well we fit known data. Backtest = train on older movies, test on newer ones (closer to real-world use).</li>
        </ol>
      </section>

      {error ? <div className="error">{error}</div> : null}
      {!error && !report ? <div className="loading">Loading report...</div> : null}
      {report ? <MetricsDashboard report={report} /> : null}
    </main>
  );
}
