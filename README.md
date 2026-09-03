# 🏎️ F1 Race Finishing Position Prediction

Predicting Formula 1 drivers' race finishing positions from historical grid position, driver form, and championship-standing features. Built as a Kaggle competition project, comparing a **regression** approach (predict exact position) against a **multi-class classification** approach (predict finish tier: Podium / Points / Finish / Tail / DNF).

## Dataset

Historical F1 race results, 1950–2022, sourced from the Kaggle competition dataset. Includes race results, qualifying times, pit stop data, and constructor/driver standings. Raw data is not committed to this repo — see `data/README.md` for download instructions.

## Repo structure

```
f1-race-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md              # dataset download instructions
├── notebooks/
│   ├── 01_eda.ipynb            # exploratory analysis + missing data
│   ├── 02_regression_pipeline.ipynb
│   └── 03_classification_pipeline.ipynb
├── reports/
│   └── figures/                # exported charts referenced below
└── submissions/
    └── final_submission.csv
```

## EDA highlights

- Grid position is the single strongest signal in the dataset — average finishing position rises almost linearly with starting grid slot, and pole sitters convert to wins at roughly 30–50% depending on the decade, climbing toward 50%+ in the most recent seasons.
- Race calendar size and driver pool have moved in opposite directions over time: races per season have roughly tripled since 1950, while unique drivers per season has fallen from 80–100 down to ~20–25, reflecting grid consolidation into fewer, more stable teams.
- From P1, drivers convert to a finish inside the top 10 the large majority of the time; that conversion rate drops off sharply and steadily the further back on the grid a driver starts.

![EDA overview](reports/figures/eda_overview.png)

## Missing data

Qualifying (`q1_ms`, `q2_ms`, `q3_ms`, `best_qual_ms`) and pit stop timing fields are missing for 60–88% of rows, concentrated almost entirely in races before 2006 — pit stop and detailed qualifying timing simply weren't recorded pre-2006, and the current three-round qualifying format itself only started that year. `rolling_avg_position` has minimal missingness (~3%), limited to drivers' first few races before a rolling window exists.

![Missing data analysis](reports/figures/missing_values.png)

## Feature engineering

| Category | Features | Rationale |
|---|---|---|
| Grid / starting position | `grid`, `grid_sq`, `grid_advantage`, `front_row`, `top5_start`, `back_marker` | Grid slot is the dominant predictor; squared and binary-threshold versions capture non-linear drop-off |
| Form / championship | `rolling_avg_position`, `form_vs_grid`, `prev_championship_position`, `prev_championship_points`, `prev_constructor_points` | Captures recent driver performance independent of where they start |
| Interaction terms | `grid_x_form`, `champ_x_grid`, `wins_x_form` | Lets the model combine starting position with current form/standing rather than treating them independently |
| Other | `rookie_proxy`, `title_contender`, `era`, `year`, `round`, `qual_available` | Context features for driver experience, championship stakes, and era-specific rule changes |

![Feature distributions by finish tier](reports/figures/feature_distributions.png)

## Models

Both pipelines use a scikit-learn ensemble (ExtraTrees + Gradient Boosting) with k-fold cross-validation (stratified for the classifier).

**Regression** — predicts exact finishing position. Feature importance is led by `front_row` (17.4%), `rookie_proxy` (13.2%), and `grid` (12.8%), with engineered interaction terms (`form_vs_grid`, `grid_x_form`) also ranking highly.

![Regression feature importance](reports/figures/feature_importance.png)

**Classification** — bins `finishing_position` into 5 classes (Podium / Points / Finish / Tail / DNF). Feature importance shifts toward raw `grid` (12.6%) and `top5_start` (12.1%) as the top predictors.

![Classification feature importance](reports/figures/feature_importance_clf.png)

## Results

- OOF and test prediction distributions track closely for both approaches, indicating no major train/test distribution shift.
- The classifier's predicted class shares roughly match the OOF class balance (e.g. ~35–38% Points-class in both), suggesting reasonable calibration rather than the model collapsing to a single dominant class.

![Prediction distribution](reports/figures/prediction_distribution.png)

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb
```

Run notebooks in order (`01` → `02`/`03`) — `03` depends on the class-binning logic introduced after the EDA step.

## Next steps

- Fold in gradient-boosted libraries (XGBoost/LightGBM) for comparison against the current pure scikit-learn ensemble.
- Handle pre-2006 qualifying/pit-stop missingness with an explicit "data unavailable era" flag rather than imputation, since the missingness is structural, not random.
- Reconcile the regression and classification approaches into a single leaderboard-ready submission strategy.

## License

MIT — see [LICENSE](LICENSE).
