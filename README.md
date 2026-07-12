# Sepsis Risk Estimator — Streamlit Demo

An interactive demo around the sepsis-prediction pipeline trained in `sepsis_prediction.ipynb`.
The user fills in **11 clinically intuitive fields**; the remaining ~199 engineered features the
model actually expects are randomly sampled from the real training distribution on every prediction.

## Files

```
webapp/
├── app.py                       # Streamlit UI
├── model_utils.py               # Core logic (loading, sampling, prediction) — unit-testable
├── requirements.txt
├── Procfile                     # for Railway / Heroku-style platforms
├── render.yaml                  # for Render.com
├── .streamlit/config.toml       # theme
├── sepsis_pipeline.joblib       # ⬅ YOU ADD THIS (from the training notebook, Section 16)
└── feature_distributions.json   # ⬅ YOU ADD THIS (from the training notebook, Section 16b)
```

**Before running or deploying**, copy your two exported artifacts from the notebook's
`/mnt/user-data/outputs/` into this folder:
- `sepsis_pipeline.joblib`
- `feature_distributions.json`

The app will show a clear error on startup if either file is missing.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Render

1. Push this folder (including your two `.joblib`/`.json` artifacts) to a GitHub repo.
2. In Render: **New → Blueprint**, point it at the repo — it will pick up `render.yaml` automatically.
   (Or: **New → Web Service**, build command `pip install -r requirements.txt`, start command
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`.)
3. Deploy. Free tier is sufficient for a demo (model file is small, no GPU needed).

## Deploy to Railway

1. Push this folder to a GitHub repo (`Procfile` is already set up for Railway).
2. In Railway: **New Project → Deploy from GitHub repo**.
3. Railway auto-detects Python and uses the `Procfile` start command. No further config needed.

## Customizing the curated inputs

Edit `CURATED_INPUTS` at the top of `model_utils.py`. Each entry needs:
- `feature_col`: the exact engineered column name from your pipeline's `feature_cols`
  (typically `<Variable>_last` for a "current reading", or a raw demographic column like `Age`)
- `widget`: `"slider"` or `"radio"`
- `min` / `max` / `default` / `step` (for sliders) or `options` (for radio, mapping display label → numeric value)

Any column in `feature_cols` not listed in `CURATED_INPUTS` is automatically randomized from
`feature_distributions.json`, so you don't need to update anything else when adding/removing inputs.

## Notes

- **This is a research/educational demo, not a clinical tool.** The risk score reflects your
  inputs *plus* randomly sampled background features consistent with the training population —
  it does not represent a real patient's actual labs/trends.
- Because most of the 210-feature vector is randomized, a single prediction can vary noticeably.
  The app averages over multiple random draws (adjustable in the UI) and shows the spread, which
  is far more informative than any single draw.
