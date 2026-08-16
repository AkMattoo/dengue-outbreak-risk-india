# Dengue Outbreak Risk in India — DS Intern Technical Assignment

Predicts the probability that an Indian state reports a **dengue outbreak** 1–3 months ahead,
using India's IDSP outbreak surveillance records (via the EpiClim dataset) combined with
Census 2011 demographics.

**Report:** [`reports/Dengue_Report.pdf`](reports/Dengue_Report.pdf) · **Live results page:** `<your-vercel-url>`

---

## Summary of findings

- **Simple baselines beat the learned models.** Smoothed climatology leads at 1 month ahead
  (AUC-PR 0.383 against a 0.100 random floor); the plain state base rate wins at 2 and 3 months
  (0.381 / 0.398). LightGBM never leads.
- **Outbreak reporting is dominated by state identity**, not by season or recent history — the
  model is substantially learning which states have strong surveillance.
- The best approaches identify roughly **3.5x more true outbreak-months than random selection**
  (precision@20% = 0.35 against a 0.10 base rate).

---

## Why this problem, and not case-count forecasting

**The data is an event log, not a time series.** EpiClim is built from IDSP *outbreak reports*:
one row per reported outbreak, not routine weekly case counts.

| | |
|---|---|
| Dengue-related records | 1,698 |
| Years covered | 2009–2022 |
| Districts appearing | 345 |
| **Median records per district** | **3** |
| Districts with ≥20 records | 17 |

A district-week forecasting model needs hundreds of observations per district. There are a
handful. Fitting case counts to this would produce confident-looking numbers with nothing behind
them.

Aggregating to **state × month** gives a complete, modellable grid:

| | |
|---|---|
| States retained | 22 of 33 (≥8 months of reported activity) |
| Months | 168 (2009-01 to 2022-12) |
| Grid cells | 3,696 |
| Cells with a reported outbreak | 18.3% |

So the question becomes: **will state *S* report a dengue outbreak in month *t+h*?** — a
probabilistic classification problem, evaluated against a seasonal-climatology baseline.

Choosing the question the data can actually answer, rather than forcing the expected one, is a
deliberate design decision. `notebooks/00_inspect_raw.py` prints the evidence for it.

### The assumption this rests on

Zero-filling the grid means treating "no row" as "no outbreak". In truth it means **no outbreak
was *reported***. States with stronger surveillance report more, so the target partly measures
surveillance capacity rather than disease. This is presence-only data with reporting bias, and no
modelling choice removes it — it is stated here, in the code, and in the report.

---

## Data

| Dataset | Source | Coverage | Used for |
|---|---|---|---|
| Dengue outbreak reports | [EpiClim (Zenodo)](https://zenodo.org/records/14580510), from [IDSP](https://idsp.mohfw.gov.in/) | 2009–2022, district-week events | Target |
| Population, density | Census of India 2011, Table 2A | 35 states/UTs | Static predictors |
| Climate *(implemented, not run)* | [Open-Meteo Archive](https://archive-api.open-meteo.com/v1/archive) | monthly, per state centroid | Lagged predictors |
| State-year cases *(context only)* | [data.gov.in](https://www.data.gov.in/search?title=dengue&type=resources&sortby=_score) | 2018–2021 | Cross-check |

EpiClim paper: [IOPscience](https://iopscience.iop.org/article/10.1088/2752-5309/ae30f6) ·
[arXiv](https://arxiv.org/abs/2501.18602). Licence CC-BY-4.0.

### Getting the data

Raw data is **not committed** (see `.gitignore`) — the code that processes it is.

1. Download `Final_data.csv` from <https://zenodo.org/records/14580510> into
   `data/raw/epiclim/EpiClim_Final_data.csv`.
2. `python -m src.prepare_population` — converts the bundled Census Table 2A extract into
   `data/raw/state_population.csv`.
3. `python -m src.derive_state_coords` — computes state centroids from EpiClim's own district
   coordinates (no manual lookup needed).
4. `python -m src.fetch_climate` — *optional*, needs internet, no API key.

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python notebooks/00_inspect_raw.py   # 1. look at the raw data before trusting anything
python -m src.prepare_population     # 2. Census 2011 -> state_population.csv
python -m src.derive_state_coords    # 3. state centroids from EpiClim
python -m src.fetch_climate          # 4. OPTIONAL: monthly climate (not run in the reported results)
python -m src.epiclim_prep           # 5. event log -> complete state x month panel
python -m src.features               # 6. lags, seasonality, climatology, targets
python -m src.train                  # 7. baselines + models, walk-forward validation
python -m src.visualize              # 8. figures

python -m pytest tests/ -q           # 20 tests, incl. the leakage check
```

Or `make all` (and `make pop coords climate` for the optional steps).

---

## Project structure

```
dengue-forecasting/
├── data/
│   ├── raw/epiclim/          EpiClim Final_data.csv (gitignored)
│   ├── raw/                  census extract, derived population + coords
│   └── processed/            panel, model table, results, predictions
├── src/
│   ├── config.py             every setting, plus the problem framing
│   ├── cleaning.py           state-name normalisation (unit-tested)
│   ├── prepare_population.py Census 2011 -> state_population.csv
│   ├── derive_state_coords.py state centroids from EpiClim coordinates
│   ├── epiclim_prep.py       stage 1: event log -> complete state x month grid
│   ├── fetch_climate.py      optional: Open-Meteo monthly climate
│   ├── features.py           stage 2: history, seasonality, climatology, targets
│   ├── train.py              stage 3: baselines + models, walk-forward validation
│   ├── evaluate.py           imbalanced-classification metrics
│   └── visualize.py          stage 4: report figures
├── notebooks/00_inspect_raw.py
├── tests/                    unit tests, including a mechanical leakage check
└── reports/                  Dengue_Report.pdf, figures, results.csv
```

---

## Modelling approach

**Target.** `outbreak ∈ {0,1}` — did the state report a dengue outbreak, 1/2/3 months ahead.

**Features** (all strictly from information available at prediction time):

| Group | Features | Rationale |
|---|---|---|
| History | outbreak and count lags 1, 2, 3, 6, 12; rolling means over 3/6/12 months; months since last outbreak | Dengue is endemic and clustered — recent activity is the strongest signal |
| Climatology | expanding state × month rate, expanding state base rate | What usually happens here at this time of year |
| Seasonality | month, sin/cos encoding, monsoon and post-monsoon flags | Outbreaks peak Aug–Oct, following the monsoon |
| Surveillance | COVID-year flag (2020–21) | Reporting collapsed then; flagging it stops the model reading disruption as a real decline |
| Static | log population, log density, decadal growth | *Aedes aegypti* breeds in stored water in dense urban settlements |
| Climate *(optional)* | temperature, rainfall, humidity at lags 1–3 plus anomalies vs the state's normal | Mosquito breeding responds to rain and heat with a 1–2 month lag |

**Baselines** — the bar any model must clear:

- **Seasonal climatology** — this state's historical rate for this calendar month.
- **Smoothed climatology** — the same, shrunk toward the state's overall rate
  (empirical Bayes, `α = 10`). Raw state × month cells rest on ~10 observations each and are
  very noisy; shrinkage keeps the seasonal signal only where evidence supports it.
- **Persistence** — did this state report last month.
- **State base rate** — the state's overall rate, ignoring season entirely.

**Models.** LightGBM (small trees, `num_leaves=15`, given ~3.7k rows at 18% positive) and
logistic regression (transparent, well-calibrated, interpretable coefficients).

**Validation.** Walk-forward with an expanding window: for each of the final
`N_TEST_YEARS = 3` years, train on everything before January of that year and predict it.
Several independent out-of-sample years, not one — which matters on a dataset this small, where a
single split could land on an unusual year and tell you nothing.
`tests/test_no_leakage.py` corrupts all data from a cutoff onward and asserts that no earlier
feature value changes.

**Metrics.** Only ~18% of cells are positive, so accuracy is meaningless (predicting "never"
scores 82%). Reported instead:

- **AUC-PR** — the right summary for a rare positive class. Read it against the base-rate floor, not 0.5.
- **Brier score** — squared error of the probability itself; rewards calibration, not just ranking.
- **Precision@20% / lift** — of the state-months flagged highest-risk, how many really reported an
  outbreak. This is the operational question: a surveillance team can only investigate so many places.
- **ROC-AUC** — for comparability, with the caveat that it flatters imbalanced data.

---

## Results

Pooled over three walk-forward held-out years. `lift` is the improvement over random selection.

| Approach | Horizon | AUC-PR | Brier | Precision@20% | Lift |
|---|---|---|---|---|---|
| Seasonal climatology | 1m | 0.348 | 0.101 | 0.325 | 3.25x |
| **Smoothed climatology** | **1m** | **0.383** | **0.089** | **0.351** | **3.51x** |
| State base rate | 1m | 0.381 | 0.088 | 0.338 | 3.38x |
| LightGBM | 1m | 0.370 | 0.097 | 0.299 | 2.99x |
| Logistic regression | 1m | 0.292 | 0.107 | 0.273 | 2.73x |
| Persistence | 1m | 0.230 | 0.125 | 0.227 | 2.27x |
| **State base rate** | **3m** | **0.398** | **0.089** | **0.345** | **3.29x** |
| Smoothed climatology | 3m | 0.351 | 0.095 | 0.338 | 3.23x |
| LightGBM | 3m | 0.342 | 0.101 | 0.303 | 2.90x |

Full table for every horizon: `data/processed/results.csv`.

**Interpretation.** No learned model beat the best baseline. This is reported rather than buried:
it indicates that outbreak *reporting* is driven mainly by which state is being observed, not by
season or recent activity. That is the reporting-bias problem showing up quantitatively, and it is
the most informative result in the project.

![Seasonality](reports/figures/01_seasonality.png)
![Data sparsity](reports/figures/02_data_sparsity.png)
![Model comparison](reports/figures/03_model_comparison.png)

---

## Assumptions and limitations

1. **Reported outbreaks ≠ outbreaks.** A zero means nothing was reported. Maharashtra, Tamil Nadu
   and Karnataka dominate the data, plausibly reflecting surveillance capacity more than burden.
2. **Presence-only data.** Negatives are assumed, never observed.
3. **Outbreak occurrence ≠ case burden.** One reported outbreak may be 5 cases or 15,000.
4. **State aggregation loses spatial detail** — but district level is far too sparse to model.
5. **2020–21 surveillance was disrupted** by COVID; flagged, but the underlying disruption cannot
   be undone.
6. **Climate is sampled at one point per state**, and that point is the centroid of *reporting*
   districts, not a true geographic centroid.
7. **Population is from Census 2011**, so it is increasingly stale across the study period.
8. **Near-silent states were dropped** (`MIN_ACTIVE_MONTHS = 8`); results do not generalise to them.
9. **Associational, not causal.**

## Next steps

- Model outbreak *size* conditional on occurrence — a two-stage hurdle model.
- Explicitly model reporting propensity per state to separate surveillance from disease.
- Spatial spillover features from neighbouring states.
- Recover finer resolution by parsing IDSP weekly PDFs directly.
- Calibrated prediction intervals; an operational system needs a plausible worst case.
