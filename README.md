# Storm Warning

**A live system that estimates, every five minutes, the probability that a geomagnetic storm begins in the next 90 minutes, warns when it is high, and scores itself and NOAA against what actually happened.**

Spacecraft at L1, about 1.5 million km sunward of Earth, measure the solar wind roughly 12–80 minutes before it arrives, depending on its speed. Storm Warning archives NOAA's real-time feeds from those spacecraft, forecasts storm onsets, compares its warnings with NOAA's own, retrains behind a statistical gate, and publishes its results, reliability and incidents.

🌐 **Website:** _link once live_ · 📊 [Backtest](docs/backtest.md) · ⚖️ [Train/serve skew](docs/skew.md) · 🧭 [Decisions](docs/decisions.md) · 🧾 [Incidents](docs/incidents.md)

---

## Results

*Test period 1 January 2023 – 30 June 2026, run once and tagged `backtest-v1`. It includes the May 2024 G5 and October 2024 G4 storms. Intervals are 90% bootstrap intervals resampling whole storms (false alarms: whole months).*

| | Storm Warning, operational inputs | Storm Warning, science inputs | NOAA K-index warnings (K ≥ 5) |
|---|---|---|---|
| Storms warned of | _pending_ | _pending_ | _pending_ |
| Median warning time (capped at 90 min) | _pending_ | _pending_ | _pending_ |
| False alarms per month | _pending_ | _pending_ | _pending_ |
| Hit rate at NOAA's false-alarm rate | _pending_ | _pending_ | — |
| Brier skill score vs climatology (90 min / 3 h) | _pending_ | _pending_ | — |

**Operational inputs** are the archived real-time feeds a live system would have seen (DSCOVR, ACE). **Science inputs** are NASA's reprocessed OMNI data, used for training. The difference between the two columns is the measured cost of train/serve skew. NOAA's warnings come from SWPC's own archive of issued messages and are scored with the same rules as the model's alerts, against two onset definitions: GFZ's Hp30, and the time NOAA's own K-index reached 5. In May 2024 those differed by nearly three hours, so a comparison is only claimed where it holds under both.

Both NOAA and this model see a storm coming only once its solar wind reaches L1, so warning times are tens of minutes. On 10 October 2026 the shock reached L1 at about 14:34 UTC, NOAA warned of K = 5 at 14:42, and K = 5 was reached at 15:35. That is the bar.

---

## How it works

```
NOAA real-time solar wind ─► archive ─► validate ─► resample ─► propagate ─► features ─► champion  ─► forecasts ─► API, site
  (SOLAR1 · ACE · IMAP)       (raw)    quarantine   (5 min)    (two clocks)   (shared)    challenger
GFZ Kp + Hp30 ─────────────────────────────────────────────────────────────────────────► score ─► drift ─► monthly gate
NASA OMNI (5–6 weeks behind) ──────────────────────────────────────────────────────────► skew report
```

| Job | When | Does |
|---|---|---|
| Archive | Every minute | Save NOAA's real-time solar wind, which NOAA keeps for 24 hours |
| Ingest and predict | Every 5 minutes | Validate, propagate, predict with champion and shadow challenger |
| Products | Every 5 minutes (positions hourly) | Save NOAA warnings, Kp estimates and forecasts, spacecraft positions; GFZ Kp and Hp30 |
| Publish | Every 5 minutes | Write website snapshots to object storage |
| Score | Every 30 minutes | Join forecasts to outcomes as indices arrive |
| Compact · drift · backup | Daily | Merge small files; check inputs and performance; back up off-site |
| Re-score · skew · retrain | Monthly | Re-score on definitive Kp; compare live spacecraft with OMNI; retrain behind the gate |

---

## Decisions that matter

**A walking skeleton came first.** By week 3 a climatology model ran end to end in production with CI/CD, monitoring and rollback. The real model replaced it through the same promotion gate as every later model.

**Training and serving data come from different spacecraft, so the skew is measured.** Training uses OMNI, which has drawn on Wind since 2020. Since 30 June 2026 NOAA's operational source has been SOLAR-1, with ACE as backup and IMAP being added. The headline backtest therefore runs the trained models on *archived operational* inputs, and a [skew report](docs/skew.md) compares the streams minute by minute. Once OMNI catches up each month, the system's own archive of live spacecraft data is compared with it.

**No density or temperature features, yet.** In the same minutes, the live spacecraft disagreed on proton density by factors of 3 to more than 10 and on temperature by nearly 6, while agreeing on speed within about 1%. v1 uses speed, magnetic field and density-free coupling functions. The skew report decides whether that changes.

**Every measurement has two clocks.** `observed_at` is when plasma was measured at L1; `arrives_at` is when it reaches Earth. A record can be used once it is observed, though it has not arrived. That gap is the lead time. OMNI's own time shift is used only to recover `observed_at`; one shared function then propagates archive and live data identically, from spacecraft position and radial speed.

**Storms are identified with Kp, timed with Hp30.** A storm is a run of 3-hour windows with Kp ≥ 5- (G1 on NOAA's scale). Kp only places it within 3 hours, so onset is timed with GFZ's half-hourly Hp30, with SYM-H as a sensitivity check.

**Only predict when a storm can start.** From each onset until 12 hours after its storm ends (or the next onset, if sooner), no new onset is possible. Those rows are excluded from training and scoring, and live the system reports "storm in progress".

**Warnings are scored as events, under the same rules for model and NOAA.** An alert switches on at a trigger level and off only after 30 minutes below it. A storm is warned of if an alert is active at onset. Credited warning time is capped at the 90-minute horizon, and alert-hours per month are reported so leaving alerts on is not free.

**Retrain monthly, promote on evidence.** Champion and candidate are only ever compared on data neither was trained on. A routine refresh must do no worse, within a small tolerance, on the newest month, held out from both. A configuration change is re-run walk-forward on identical folds over the latest 24 months and must be better with bootstrap probability ≥ 0.9. Candidates run in shadow first. Every decision is published, including rejections.

**Reliability has numbers.** Two SLOs, each 99.5% a month: a forecast record within 2 minutes of every 5-minute slot, and an available API. Error budgets are on the system page. Upstream data availability is reported but is not an objective, because the spacecraft are outside the system's control.

The full reasoning is in the [decisions log](docs/decisions.md).

---

## Stack

Python 3.13 · uv · Polars · DuckDB · LightGBM · scikit-learn · pandera · MLflow (registry aliases) · FastAPI · APScheduler 3 · Prometheus · Grafana Cloud · healthchecks.io · Docker Compose · Caddy · GitHub Actions · GHCR · Astro

Parquet on disk with a SHA-256 manifest of every raw file; no database server. Polars runs the pipelines; scoring, metrics and drift are SQL files in [`sql/`](sql/), run by DuckDB. Everything except the website runs on one small always-on server.

---

## Running it locally

Requires [uv](https://docs.astral.sh/uv/) and, for the full service, Docker.

```bash
git clone <repository-url>
cd stormwarn
uv sync --locked

uv run stormwarn backfill                      # OMNI 1-minute, GFZ, archived NOAA data (~5 GB download)
uv run stormwarn labels
uv run stormwarn train --horizon 90m
uv run stormwarn backtest --inputs operational # writes docs/backtest.md and figures
uv run stormwarn skew-report                   # writes docs/skew.md
uv run stormwarn scheduler                     # run the live jobs

uv run pytest -m "not live" --cov=stormwarn
uv run ruff check
```

Full service:

```bash
cp .env.example .env              # set GH_OWNER, IMAGE_TAG, image versions and secrets
docker compose up -d
curl localhost:8000/forecast/latest
```

### API

| Endpoint | Returns |
|---|---|
| `GET /health` | `ok`, `degraded` or `stale`; data age per feed; active spacecraft; last success per job; model versions |
| `GET /forecast/latest` | Probability of onset within 90 minutes and 3 hours, alert state, or "storm in progress"; predicted Kp; NOAA's latest warning and forecast; model version; data age |
| `GET /forecast/history?hours=24` | Recent forecasts (up to 168 hours) |
| `POST /predict` | Online inference on supplied solar wind data, through the same feature code and model as the batch path |
| `GET /model` | Champion and challenger versions; gate decisions |
| `GET /metrics` | Prometheus metrics |

---

## Deployment

Every push to `main` that passes CI builds an image tagged with its commit SHA, pushes it to GHCR and deploys it over SSH. If the new version is not healthy within three minutes, the previous image is redeployed automatically. Rolling back a model is one command (`stormwarn model rollback`), which moves the registry's `@champion` alias back. A daily canary checks every upstream endpoint against its data contract and opens an issue when one changes. Runbooks are in [`docs/runbooks.md`](docs/runbooks.md).

---

## Repository layout

```
src/stormwarn/
├── ingest/          # NOAA SWPC, GFZ, OMNI, archived operational data
├── schemas.py       # data contracts (pandera)
├── validate.py      # sentinels, ranges, gaps, freshness, quarantine
├── resample.py · propagation.py · features.py   # shared by training and serving
├── labels.py        # storm events, Hp30 onsets, risk set
├── alerts.py        # alert logic, applied to the model and to NOAA
├── train.py · calibrate.py · evaluate.py · bootstrap.py
├── registry.py · gate.py · monitor.py
├── publish.py · backup.py · scheduler.py · api/main.py
sql/                 # DuckDB queries
deploy/              # deploy script, Caddy and Prometheus config
site/                # Astro website
docs/                # data, backtest, skew, decisions, runbooks, incidents
```

---

## Status

- [ ] **Week 1** — server, live archiving, backups, week-1 checks
- [ ] **Weeks 2–3** — walking skeleton: baseline model live with CI/CD, rollback, monitoring and SLOs
- [ ] **Weeks 4–6** — backfill, labels, features, leakage and parity tests
- [ ] **Weeks 7–9** — model, science and operational backtests, skew report
- [ ] **Weeks 10–12** — live scoring, drift, shadow challenger, gated retraining, restore drill
- [ ] **Weeks 13–16** — website and polish

### Known limitations

- Lead time is bounded by how far upstream L1 is; the fastest storms give the least notice.
- SOLAR-1 has been NOAA's operational source only since June 2026 and is not yet independently validated; the skew report tracks it.
- Onsets are located to 30 minutes, and Hp30 has no definitive version.
- NOAA warns about its own K-index, which can reach storm level hours after Hp30 does; model-versus-NOAA claims are made only where they hold under both definitions.
- Solar Cycle 25 peaked in October 2024. Live storms will be few for several years, so the backtest carries the headline.

---

## Data and credits

- **Solar wind archive:** NASA/GSFC Space Physics Data Facility, high-resolution OMNI data (King and Papitashvili, 2005).
- **Kp and Hp30:** GFZ Helmholtz Centre for Geosciences, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Matzka et al. (2021), *Space Weather*, doi:10.1029/2020SW002641, and dataset doi:10.5880/Kp.0001; Yamazaki et al. (2022), *GRL*, doi:10.1029/2022GL098860, and Hpo dataset V3.0, doi:10.5880/Hpo.0003.
- **Real-time solar wind, Kp estimates, warnings and forecasts, and the archive of issued alerts since 2002:** NOAA Space Weather Prediction Center.
- **Archived operational solar wind:** NOAA NCEI (DSCOVR) and NASA SOHO archive (ACE real-time data).

## Licence

MIT
