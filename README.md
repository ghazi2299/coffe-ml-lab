# Coffee Fermentation Lab

Web app that learns from fermentor sensor logs. It does two things and compares tabular machine learning with deep learning on time series:

- **Mode 1, process parameter forecasting:** pH, CO₂, O₂, temperature, pressure and humidity from 30 minutes up to 6 hours ahead.
- **Mode 2, cup quality prediction:** the cupping score of a batch, rated acceptable or below standard.

**Models.** Tabular: Ridge, PLS, SVR, Random Forest, Extra Trees, Gradient Boosting, Gaussian Process (Mode 2 uses all, Mode 1 a subset). Deep learning: GRU, LSTM, TCN (Mode 1) and GRU, LSTM, 1D-CNN (Mode 2).

FastAPI backend (scikit-learn and PyTorch) with a dependency-free static frontend (vanilla JS and SVG, self-hosted fonts). The UI is light-only by design.

## Run locally
```bash
pip install -r requirements.txt
uvicorn app:app --reload          # http://127.0.0.1:8000
```

## Using the app
1. Pick **Mode 1** or **Mode 2**, then a data source: upload files or the bundled sample.
2. Choose a preset (**Standard** or **High accuracy**) or open **Advanced settings** and change anything: which models run, number of trees and depth, boosting rate, epochs, ensemble seeds, learning rate, hidden units, dropout, batch size, validation folds and repeats, forecast horizons, look-back window, random seed and more. Editing a value switches the preset to *Custom*.
3. **Show charts and trends** and **Show full technical details** read the finished run; nothing is trained again. Technical details cover data, parameters, comparison, training and validation, plus phases and forecast values (Mode 1), or feature importance, batch clusters and final-model predictions (Mode 2). Both modes have an Actual vs Predicted tab (values, error and charts per model) and CSV download on the larger tables. Results saved before this version need one retraining to fill the Mode 1 validation table.
4. **Show Predict on New Data** (both modes) scores new sensor logs with the already trained model, with no retraining. Mode 1 forecasts the next 6 hours from the end of each uploaded run and, where the file already contains the future, shows actual vs predicted with the error. Mode 2 scores each batch and, with an optional cupping file, shows actual vs predicted with the error.
5. **Combine ML** merges both models into one stop-time advisory.

### Sensor logs (both modes)
`.csv` files or a `.zip` of fermentor logs. All files are merged and de-duplicated by timestamp, split into batches at time gaps longer than 60 min, and runs shorter than 2 h are dropped.

### Mode 2: sensor logs and cupping scores are separate files
Mode 2 takes the **same sensor logs as Mode 1** plus an optional, separate **cupping file**:

```
session,score
1,82.5
2,79.0
...
```
`session` = 1, 2, 3… in time order of the batches. Optional `start,end` columns define the batch windows explicitly. An example is at `GET /api/lab/cupping/example.csv`.

| Situation | What happens |
|---|---|
| Several batches and a cupping file (at least 6 scored batches) | Trains on the real scores. |
| One batch only (or fewer than 6) | The run is split into consecutive pseudo batches. Scores are random, test only. |
| No cupping file | Still runs, with random scores. Test only, metrics are meaningless. |

The result page states which case was used, and the combined advisory warns when the cup model came from test data.

## Project layout
```
app.py              FastAPI entry point (security headers, static files, /healthz)
fermentlab/
  web.py            API routes, background training jobs
  settings.py       settings schema, presets and validation (the UI is built from this)
  batches.py        Mode 2: batches + cupping file, pseudo split, random test scores
  data.py           log reading and cleaning
  windows.py        forecasting windows        features.py   batch features
  forecast.py       Mode 1 models              cupping.py    Mode 2 models, clustering
  seqnets.py        GRU / LSTM / TCN / 1D-CNN  cv.py         validation splits
  phases.py         fermentation phase detection
static/             index.html, lab.js, lab.css, styles.css, fonts
data/               sample_fermentor.csv
scores_template.csv example cupping file with start,end windows
```

## API
| Route | Purpose |
|---|---|
| `GET /api/lab/settings` | settings schema and presets |
| `POST /api/lab/forecast/train` | Mode 1 training (`files`, `source`, `quality`, `settings` JSON) |
| `POST /api/lab/cupping/train` | Mode 2 training (`files`, optional `cupping`, `source`, `quality`, `settings` JSON) |
| `GET /api/lab/status/{forecast\|cupping}` | job progress and result |
| `POST /api/lab/cupping/predict` | score new batches with the trained Mode 2 model |
| `POST /api/lab/merge` | combined advisory from both models |

## Deploy
Any host that can run a Docker container works. The app listens on `$PORT` (default 8000) and exposes `/healthz`.
**Render:** New → Blueprint → select the repo (`render.yaml` included), plan with at least 1 GB RAM.
Keep a **single worker**: training jobs and trained models are held in process memory.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | 8000 | listening port |
| `MAX_UPLOAD_MB` | 50 | upload size limit per request |
| `FERMENT_MODELS` | `./models` | where trained models are saved |

## Train locally, serve from saved results
1. Locally (same package versions as `requirements.txt`): `uvicorn app:app`, upload `Fermentor_clean.zip`, train.
2. Training writes to `models/`: `forecaster.joblib`, `cupping_model.joblib`, `result_forecast.json`, `result_cupping.json`, `last_run.csv`, `state.json`. Commit the whole `models/` folder.
3. On start the app loads saved results automatically and shows them. Training stays available.
4. Optional: set `SERVE_ONLY=1` to lock training and show results only.
