# Will This Movie Make Money? — Predict movie profitability from metadata

I will be experimenting with movies and their box office collections to understand profitability, what parameters box-office success depends on, and make it interactive for you to play with predictions of upcoming movies.

## Quickstart

1. Clone the repo:
   git clone https://github.com/swa311096/will-this-movie-make-money.git
   cd will-this-movie-make-money

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows (PowerShell)

3. Install dependencies:
   pip install -r requirements.txt

4. Run a quick local prediction:
   python src/predict.py --title "Example Movie" --budget 10000000

Expected example output:
```json
{"title":"Example Movie","budget":10000000,"predicted_profitable":false,"probability":0.18}
```

## V1 data + model pipeline (market-wide bootstrap)

This repository now includes a first-pass historical pipeline designed to:
- pull released movies (all studios) from Box Office Mojo year charts within a date window,
- crawl each movie's domestic daily table,
- build a training table with early-run features,
- train two baseline regressors:
  - domestic multiplier (`final_domestic / opening_weekend`)
  - international/domestic ratio (`final_international / final_domestic`)

### Build training data

```bash
playwright install chromium
python3 scripts/build_training_data.py --years 3
# if requests get blocked on some pages:
python3 scripts/build_training_data.py --years 3 --use-playwright
```

Outputs are stored in:
- `data/processed/boxoffice.sqlite` (raw + transformed tables)

### Train baseline models

```bash
python3 scripts/train_baseline_models.py
# optional: disable outlier filters
python3 scripts/train_baseline_models.py --no-filters
```

Saved models:
- `data/models/domestic_multiplier_rf.joblib`
- `data/models/intl_dom_ratio_rf.joblib`

### Evaluate current models

```bash
python3 scripts/evaluate_models.py
# optional: write report directly for the dashboard
python3 scripts/evaluate_models.py --output-json ui/public/evaluation_report.json
```

Evaluation report includes:
- in-sample metrics on filtered rows,
- chronological backtest metrics (time split),
- chart-ready monthly MAE aggregates.

## TSX metrics dashboard

```bash
cd ui
npm install
npm run dev
```

Dashboard reads: `ui/public/evaluation_report.json`.
Refresh it after retraining by running:

```bash
python3 scripts/evaluate_models.py --output-json ui/public/evaluation_report.json
```

## What this project is for

- Collect and clean historical box-office and movie metadata.
- Explore which features most influence profitability (budget, genre, cast, release date, runtime, ratings, marketing).
- Train and evaluate baseline and advanced models to predict profit/loss or revenue.
- Provide an interactive demo (CLI or small web app) so others can try predictions for upcoming movies.

## Project layout (minimal)

- src/          # package code (data ingestion, features, models, CLI)
- models/       # saved trained models (use Git LFS for large files)
- data/         # data downloads or link manifests (do NOT commit large raw files)
- tests/        # unit tests (optional)
- .github/      # workflows (optional)

## Data & models

- Add raw datasets to data/ or document download commands in data/README.md.
- Do not commit large raw datasets or model artifacts; use Git LFS or external storage and document locations.

## Contributing

- Fork and open PRs.
- Run tests locally: pytest
- Follow code style (black, flake8). Add a CONTRIBUTING.md if you want contributor guidelines.

## License

MIT — see LICENSE
