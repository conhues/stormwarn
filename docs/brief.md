# Storm Warning — Project Brief

*Python package: `stormwarn`. Sources and endpoints verified 12–13 October 2026; see §3.2 for what was and wasn't confirmed.*

A live, self-running system that estimates every five minutes the probability that a geomagnetic storm begins in the next 90 minutes, compares its warnings with NOAA's, scores itself as outcomes arrive, and retrains behind a statistical gate. It is a portfolio project whose purpose is to demonstrate **production machine learning engineering**. The modelling is careful, but the system around it is the point.

---

## 1. Summary

### 1.1 What it does

Every five minutes the system:

1. Polls NOAA's real-time solar wind feeds and archives every record, because NOAA keeps only the last 24 hours.
2. Validates the batch, quarantining anything that fails, and propagates each measurement from the spacecraft to Earth.
3. Publishes the probability of a storm onset in the next 90 minutes (and in the next 3 hours), or reports that a storm is already in progress.
4. Runs a shadow challenger model alongside the live one and logs both.

On slower schedules it scores forecasts against GFZ's indices, checks for drift and train/serve skew, retrains monthly behind a promotion gate, backs itself up, and publishes snapshots for a static website.

### 1.2 What it demonstrates

| MLE competency | Where it shows up |
|---|---|
| Walking skeleton, then iterate | A baseline model is live end to end by week 3; the real model replaces it through the same gate as every later model (§10) |
| Point-in-time correctness | Two clocks per record, simulated publication delays, a leakage test (§3.6, §8.6) |
| Train/serve skew, measured | Training uses NASA's science-grade OMNI data; serving uses NOAA's operational feeds from different spacecraft. A skew report and a second backtest on archived *operational* data quantify the gap (§4.4) |
| Data contracts | Schema and range validation on every feed, quarantine, a daily canary against live endpoints (§6.5, §8.6) |
| Model lifecycle | MLflow registry with aliases, shadow challenger, two-track promotion gate, one-command rollback (§5) |
| CI/CD | Tests and lint on every push; image built and pushed to GHCR by commit SHA; automated deploy with health check and automatic rollback (§8.5) |
| Observability | Prometheus metrics, SLOs with error budgets, alerting, a dead-man's switch, an incident log and runbooks (§7) |
| Reliability | Atomic writes, graceful degradation, nightly off-site backups with a tested restore (§6.7, §7.6) |

### 1.3 The headline result

> "On archived operational data from 2023 to mid-2026, the model warned of X% of storms with a median of Y minutes' notice, at Z false alarms per month. NOAA's K-index warnings over the same storms warned of X′% with a median of Y′ minutes."

That sentence is filled in from the **operational backtest** (§4.3), not the science-data backtest, because the operational one uses the inputs a live system would actually have seen. NOAA's figures come from SWPC's own archive of issued warnings (§3.1), scored by the same rules. The comparison is only claimed where it holds under both onset definitions in §2.7.

### 1.4 Expectations

- **Lead time is bounded by physics.** Monitoring spacecraft sit at L1, about 1.5 million km sunward. Plasma takes roughly 80 minutes to reach Earth at 300 km/s, 30–60 minutes at 400–800 km/s, and under 20 minutes at the 1,500 km/s or more of the fastest eruptions. Median warning times will be in the tens of minutes.
- **NOAA's forecasters use the same data.** On 10 October 2026 the shock reached L1 at about 14:34 UTC; at 14:42 SWPC warned that K = 5 was expected, and K = 5 was reached at 15:35. Matching that is a good result. Beating it consistently would be surprising, and the documentation should say so rather than oversell.
- **Most retrains will not be promoted.** That is the gate working.

### 1.5 Scope

**In (v1):** onset probability at 90-minute and 3-hour horizons; next-window Kp; science and operational backtests; skew report; live scoring and re-scoring; drift checks; shadow challenger; gated monthly retraining; CI/CD with rollback; Prometheus, SLOs and alerting; backups; an Astro website.

**Out:** aurora visibility, survival models, forecasting from solar imagery (days ahead), Kubernetes, multi-region anything.

---

## 2. Prediction problem

### 2.1 Definitions

| Term | Definition |
|---|---|
| Kp | GFZ's 0–9 planetary index for fixed 3-hour UTC windows, stored in thirds (5- = 4.667, 5o = 5.000, 5+ = 5.333). |
| Hp30 | GFZ's half-hourly Kp-like index. Open-ended: it exceeded 11 during May 2024. |
| Storm window | A 3-hour window with Kp ≥ 4.66. The tolerance covers GFZ's 4.667 and SWPC's 4.67. SWPC's own 3-day forecast labels 4.67 as G1, so this matches NOAA's G-scale. |
| Storm event | A run of storm windows; runs less than 12 hours apart merge. The event ends at the end of its last storm window. |
| Onset | Let *W* be an event's first storm window and *W*₋₁ the window before. The onset is the start of the first Hp30 interval with Hp30 ≥ 4.66 between the start of *W*₋₁ and the end of *W*. If none exists, it is the start of the highest Hp30 interval in *W*. |
| At risk | Time *t* is at risk unless, for some event *k*, onsetₖ ≤ *t* < min(endₖ + 12 h, onsetₖ₊₁). **The exclusion starts at onset, not at the start of *W*,** otherwise an onset inside *W*₋₁ would leave storm-time rows labelled 0. It is cut short at the next onset, because an onset in the next event's *W*₋₁ can fall less than 12 hours after this event ends, and that storm would otherwise have no positive rows. |
| Prediction time *t* | A 5-minute slot boundary when a forecast is made. |
| Label *yₕ(t)* | Defined only for at-risk *t*: 1 if an onset falls in (*t*, *t* + *h*], else 0. |

### 2.2 Horizons

| Horizon | Role | Why |
|---|---|---|
| **90 minutes** | Primary. Drives alerts and warning-time scoring. | Covers the L1-to-Earth transit time at almost all solar wind speeds, with a margin for Hp30's 30-minute resolution. Positives further out are mostly unpredictable from L1 data, so a longer horizon would dilute skill and reward "disturbed conditions persist" instead. |
| **3 hours** | Secondary. Reported for probability skill. | Matches the Kp window and Chakraborty and Morley (2020), so results can be placed against the literature. |

One training pipeline takes the horizon as a parameter. Skill is also reported by time to onset (0–30, 30–60, 60–90 and 90–180 minutes), which shows directly where the L1 information runs out.

### 2.3 Outputs

1. **P(onset within 90 minutes)** and **P(onset within 3 hours)**, or the status "storm in progress".
2. **Predicted Kp for the next 3-hour window**, which gives a scoring signal every three hours, storm or not.

### 2.4 "Storm in progress" live

Live, the system does not know the onset until GFZ publishes the index values that reveal it. It therefore applies a separate, documented rule using only published values: *storm in progress* if the most recent published Hp30 ≥ 4.66, or the most recent published Kp window ≥ 4.66, or less than 12 hours have passed since the last such value.

Between a true onset and the first index value that reveals it, the system still publishes probabilities. In scoring, those rows fall outside the at-risk set and are excluded from per-row metrics. They are counted and reported as "forecasts made during an undetected storm". The features include the latest published Hp30 and Kp and the time since the last storm, so the model sees the same information the live rule does.

### 2.5 From probabilities to warnings

The model predicts probabilities. Warnings are derived from them and scored as events:

1. **Trigger.** An alert starts when P₉₀ ≥ τ. It ends when P₉₀ has stayed below τ for 30 consecutive minutes. The debounce stops a probability hovering around τ from producing a string of alerts. τ is chosen on the validation folds only.
2. **Hit.** A storm is warned of if an alert is active at its onset.
3. **Warning time.** Onset − max(alert start, onset − 90 min). The cap stops an alert that happened to start six hours early from scoring six hours of warning.
4. **False alarm.** An alert that starts at an at-risk time and ends with no onset between its start and 30 minutes after its end. False alarms whose peak Kp reached 4 are reported separately as near misses.
5. **Cost of alerting.** Alert-hours per month, which catches a model that "warns" by leaving alerts on.

**The same rules are applied to NOAA.** A NOAA warning is a K-index warning of K ≥ 5 (message codes WARK05, WARK06, WARK07) issued at an at-risk time. It is active from its issue time to its "Valid To" time, extended by any later "Now Valid Until", or until it is cancelled if that comes first. NOAA's warnings are scored against GFZ onsets, not against NOAA's own Kp estimates.

**Onset timing bias.** Onsets are dated to the start of their Hp30 interval, so the true onset can be up to 30 minutes later and warning times are understated by up to 30 minutes, equally for the model and NOAA. The backtest also reports warning times with onsets at interval midpoints.

### 2.6 Baselines and comparators

| Comparator | Compared on | Availability |
|---|---|---|
| Climatology | Probabilities (Brier skill score reference) | Onset base rate over the trailing year of training data |
| Persistence | Next-window Kp | Trivial |
| NOAA K-index warnings | Hit rate, warning time, false alarms | SWPC's alerts archive, 2002 to last month (verified); live from `alerts.json` |
| NOAA 3-day forecast | Next-window Kp | NCEI archive from March 2022 (verified), and live |

Wing Kp is **not** a comparator. SWPC stopped ingesting it in June 2018 and replaced it with the Geospace model. SWPC's Geospace predicted-Kp file has not updated since June 2024, so it is not used either.

### 2.7 Onset definitions and a fair NOAA comparison

NOAA warns that *its* K-index will reach 5; this project times onsets with GFZ's Hp30. The two can disagree by hours, and May 2024 shows it:

| Time (UTC, 10 May 2024) | Event |
|---|---|
| 12:00–15:00 | GFZ Kp 3.667 (not a storm window) |
| 14:30 | First Hp30 ≥ 4.66 (5.667), staying above it through the storm, so the Hp30 onset is 14:30 |
| 16:39 | Interplanetary shock observed at L1 (SWPC `WARSUD`) |
| 16:51 | SWPC warns that K = 5 is expected, valid from 17:00 (`WARK05`) |
| 17:05 | Sudden commencement at Earth |
| 17:14 | SWPC alerts that K = 5 was reached in the 15–18 period, with "Active Warning: Yes" (`ALTK05`) |

Under the Hp30 onset, NOAA's warning came 2 hours 21 minutes late, a miss. By NOAA's own index it was a hit. Scoring NOAA only against Hp30 onsets would flatter the model for the wrong reason. So every model-versus-NOAA result is computed twice:

1. **Hp30 onset** (§2.1): the project's primary target, applied to both.
2. **NOAA-aligned onset:** the time NOAA's K first reached 5 in the event: the "Threshold Reached" time of the first `ALTK05`–`ALTK09` alert where the message has one (recent messages do), otherwise its issue time, which lags by minutes. Archived 2024 alerts carry only the issue time.

A claim that the model beats, matches or trails NOAA is made only if it holds under both definitions; otherwise both results are reported and the disagreement explained. The archive's "Active Warning" flags also give NOAA's own hit record, which cross-checks the warning parser.

The shock reports (`WARSUD`/`SUMSUD`, "IP Shock Passage Observed") are used to split results into shock-driven storms and others (such as high-speed streams), and to count onsets that precede any shock at L1. In May 2024 the Hp30 onset preceded the shock at L1 by over two hours; neither NOAA nor a model watching for the shock could have warned of it.

---

## 3. Data

### 3.1 Sources

| Purpose | Source | Format notes |
|---|---|---|
| **Training solar wind** | NASA OMNI high-resolution, `https://spdf.gsfc.nasa.gov/pub/data/omni/high_res_omni/` | Yearly ASCII: `omni_min{YYYY}.asc` (about 150 MB a year) and `omni_5min{YYYY}.asc` (about 33 MB). Format: `hroformat.txt` in the same directory. 46 words per 1-minute record. **Use the 1-minute files**, and resample them to 5 minutes with the same function used for live data. |
| Training solar wind, alternative access | CDAWeb HAPI, `https://cdaweb.gsfc.nasa.gov/hapi/data?id=OMNI_HRO_1MIN&...` | CSV; parameter names differ from the ASCII columns (e.g. `flow_speed`, `BZ_GSM`, `Timeshift`). Useful for spot checks. |
| **Kp history and nowcast** | GFZ, `https://kp.gfz.de/app/files/Kp_ap_since_1932.txt` and `Kp_ap_nowcast.txt` (last 30 days) | Lines starting `#` are headers; columns `YYY MM DD hh.h hh._m days days_m Kp ap D`. `D` = 1 definitive, 0 preliminary. Missing: Kp −1.000, ap −1. The nowcast file contains rows for intervals not yet computed (filled with −1) and revises values in place. |
| **Hp30 history and nowcast** | GFZ, `https://kp.gfz.de/app/files/Hp30_ap30_complete_series.txt` (from 1985) and `Hp30_ap30_nowcast.txt` (last 30 days) | Same layout as Kp. **No definitive flag**: GFZ says `D` is always 0 and reserved. Values may be revised. |
| **Live indices** | GFZ JSON API, `https://kp.gfz.de/app/json/?start=…Z&end=…Z&index=Kp` (or `Hp30`) | Parallel arrays `datetime` (interval start, UTC), the index, and `meta`. For Kp there is also `status` (`def`/`pre`); Hp30 has none. Preferred for live polling. |
| **Training labels, cross-check** | SYM-H, in OMNI (from WDC Kyoto) | Provisional; lags real time by about 4–6 weeks. |
| **Live solar wind, plasma** | NOAA SWPC, `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` | Array of objects, newest first, last 24 hours. Keys include `time_tag`, `active`, `source`, `proton_speed`, `proton_density`, `proton_temperature`, `proton_vx_gse`, `overall_quality`. `time_tag` has no zone suffix and is treated as UTC. |
| **Live solar wind, field** | NOAA SWPC, `.../json/rtsw/rtsw_mag_1m.json` | As above. Keys include `bt`, `bx_gsm`, `by_gsm`, `bz_gsm`, `overall_quality`. |
| **Live spacecraft positions** | NOAA SWPC, `.../json/rtsw/rtsw_ephemerides_1h.json` | Hourly, per spacecraft. Keys include `x_gse`, `y_gse`, `z_gse` (values around 1.5 million, so presumably km; units undocumented). |
| Live Kp estimates (logged only) | `.../products/noaa-planetary-k-index.json`; `.../json/planetary_k_index_1m.json` | Array of objects. Note `Kp` (capital K) in the first file. |
| Live NOAA Kp forecast | `.../products/noaa-planetary-k-index-forecast.json`; `.../text/3-day-forecast.txt` | Forecast JSON uses lower-case `kp`, plus `observed` (`observed`/`estimated`/`predicted`) and `noaa_scale`. Text forecast issued at 00:30 and 12:30 UTC. |
| Historical NOAA Kp forecast | NCEI, `https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast/` | `YYYY/MM/yyyymmdd0030three_day_forecast.txt` and `…1230…`, March 2022 onward. |
| **Live NOAA warnings** | `.../products/alerts.json` | Array of `{product_id, issue_datetime, message}`, newest first, about 30 days. K-index warnings have `product_id` K04W–K07W (message codes WARK04–WARK07). |
| **Historical NOAA warnings** | SWPC FTP, `ftp://ftp.swpc.noaa.gov/pub/alerts/` | `alerts_YYYY.zip` for 2002–2014; one HTML file per month, `alerts_YYYYMM.html`, from January 2015. Each closed month is written at 00:46 UTC on the 1st; the current month's file is rewritten hourly. Messages are separated by `<hr>`, with one `Label: value` per line ending `<br>`: the same text as `alerts.json`. Also holds `WARSUD`/`SUMSUD` messages with "IP Shock Passage Observed", which time interplanetary shocks at L1. Only reachable with an FTP client (the HTTPS mirror disallows automated fetching). |
| **Archived operational solar wind, DSCOVR** | NCEI, `https://www.ngdc.noaa.gov/dscovr/data/YYYY/MM/` | Daily netCDF, `oe_f1m_dscovr_…_pub.nc.gz` (plasma) and `oe_m1m_…` (field), from 26 July 2016. Collected during real-time operations and **not reprocessed**, so it is what SWPC served at the time. Newer paths redirect to NOAA's cloud archive. |
| **Archived operational solar wind, ACE** | `https://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/` | `YYYYMMDD_ace_swepam_1m.txt` and `YYYYMMDD_ace_mag_1m.txt`, August 2001 onward, still updated daily. Status column `S`: 0 nominal, 1–8 bad, 9 no data. Missing −9999.9. |

All URLs live in `config.py`. A failed or changed endpoint is an incident, not a crash.

### 3.2 Endpoint check (12–13 October 2026)

Checked with a web-fetch tool that summarises responses and sometimes serves cached copies. Structures and field names are reliable; the **ages of the newest records are not**, so real latency is re-measured from the server in week 1 (gate G2).

| Endpoint | Result |
|---|---|
| `json/rtsw/rtsw_wind_1m.json`, `rtsw_mag_1m.json` | Working. Sources seen: `SOLAR1` (`active: true`), `ACE`, `IMAP`. No DSCOVR. |
| `json/rtsw/rtsw_ephemerides_1h.json` | Working, but the newest record seen was 25 September 2026. Stale or cached: gate G3. |
| `json/rtsw/` directory listing | Server error on every attempt. |
| `products/solar-wind/*.json` (old 7-day, 1-day files) | 4xx errors. The listing shows them frozen since 25 March 2026. **Do not use.** |
| `products/noaa-planetary-k-index.json` | Working. Oldest first, about 7 days. |
| `products/noaa-planetary-k-index-forecast.json` | Working. About 7 days back and 3 days ahead. |
| `json/planetary_k_index_1m.json` | Working. Keys `time_tag`, `kp_index`, `estimated_kp`, `kp`. |
| `products/alerts.json` | Working. About 30 days; codes WARK04–07, ALTK04–07, WATA20/30 seen. |
| `text/3-day-forecast.txt` and NCEI archive | Working. Archive files for every day of 1–14 May 2024 present. |
| `json/geospace/geospace_pred_est_kp_1_hour.json` | Responds, but every record is from 18 June 2024. **Do not use.** |
| GFZ Kp and Hp30 text files, both `/app/files/` and `/fileadmin/files_for_gfz_cms/` | Working. |
| GFZ JSON API | Working; Kp through the latest 3-hour window, Hp30 through the latest half-hour. |
| OMNI directory and `hroformat.txt` | Working. Data runs to about 2 September 2026: OMNI trails real time by 5–6 weeks. |
| CDAWeb HAPI `OMNI_HRO_1MIN` | Working. |
| sohoftp ACE daily archive | Working; the 12 October 2026 file was created the same day. |
| NCEI DSCOVR archive | 2016 listing works; newer paths redirect to a JavaScript app. Download route confirmed in week 4. |
| `ftp.swpc.noaa.gov/pub/alerts/` | **Working via FTP client** (13 October): yearly zips 2002–2014, monthly HTML from `alerts_201501.html` to `alerts_202610.html`. `alerts_202405.html` parsed cleanly into 294 messages, including every K-index warning, extension and alert of the 10–11 May 2024 storm. The HTTPS mirror is closed to automated fetching. |

### 3.3 What the checks taught

- **The operational source changed in 2026.** SOLAR-1 (formerly SWFO-L1, launched 24 September 2025; `SOLAR1` in the feed) became SWPC's primary real-time source on 30 June 2026, when DSCOVR processing stopped. ACE is the backup until NASA's IMAP has 24/7 real-time coverage. The `source` field names the spacecraft and `active` marks the one in operational use; SWPC can switch at any time. **Filter on `active`, never on a spacecraft name.**
- **Spacecraft disagree on density and temperature, not speed.** In a manual check on 11 October, densities in the same minutes were about 5–9 cm⁻³ from SOLAR1, 0.1–1.7 from ACE and 17–28 from IMAP. On 12 October SOLAR1 and IMAP differed by a factor of about three in density and nearly six in temperature, with speeds within 1%. ACE reported density 0.0 with a nominal quality flag. **v1 uses no density or temperature features** (§3.7), until the skew report shows otherwise.
- **Quality flags carry little information.** SWPC says the new display does no quality flagging; `overall_quality` was 0 throughout. Validation does its own range and consistency checks.
- **NOAA's two Kp files can disagree mid-storm.** In the 11 October check, for 15:00 UTC on 10 October, `noaa-planetary-k-index.json` gave 7.00 while the forecast file's *estimated* value was 3.67. GFZ's preliminary value was 7.333: the forecast file's estimate was stale. Ground truth comes from GFZ.
- **Formats drift.** Key case differs between files (`Kp` vs `kp`), alert labels vary ("Noaa Scale" vs "NOAA Scale", "Warning Condition" vs "Warning Conditions", "K-index" vs "K-Index", "Active Warning: Yes" vs "YES"), older alerts lack the "Threshold Reached" line, and one GFZ header states 30 header lines where 29 exist. Parsers skip `#` lines rather than counting them, normalise keys, and match labels case-insensitively. Unknown keys raise a schema-drift incident.
- **Sentinels.** `null` in SWPC JSON, and `-9999` in `max_data_flag` (11 October check); `-1.000`/`-1` in GFZ files; in OMNI, by column, 99 (spacecraft ID), 999999 (time shift), 9999.99 (field, positions), 99999.9 (speeds), 999.99 (density), 9999999. (temperature) and 99999 (SYM-H and other indices); in ACE archive files, −9999.9 for density and speed and −1.00e+05 for temperature.

### 3.4 Training data

- **Period:** 1998 to the end of the test period (§4.1).
- **Spacecraft:** OMNI records whose spacecraft was upstream at L1, selected by **position**, not ID: spacecraft X (GSE) ≥ 200 Earth radii (OMNI words 32–34). OMNI's IDs are 71 = ACE, 51/52 = Wind, 60 = Geotail, 50 = IMP 8; DSCOVR is never used in OMNI. Wind did not sit at L1 throughout this period, which is why position is the filter.
- **Who supplied the data:** mostly ACE early on and essentially only Wind since 2020. **None of these is the spacecraft the live system reads.** §4.4 deals with that.
- **Pinning:** OMNI rewrites past years (2024 was rewritten in November 2025). Every raw download is stored with its SHA-256 in `data/manifest.parquet`. Training runs log the manifest hash.

### 3.5 Two clocks and one propagation function

| Time | Meaning | OMNI | Live |
|---|---|---|---|
| `observed_at` | When the plasma was measured at the spacecraft | Record time − `Timeshift` (word 10, seconds) | `time_tag` |
| `arrives_at` | Estimated arrival at the bow-shock nose | `propagate(observed_at, x_sc, vx)` | `propagate(observed_at, x_sc, vx)` |

OMNI's time tag is the arrival time at the bow-shock nose and marks the **start** of the averaging bin, and its time shift is defined as arrival time minus observation time. Subtracting it therefore recovers the observation time. OMNI's shift itself is then discarded.

`propagate` is **one function** shared by training and serving: delay = (x_sc − x_bsn) / |vx|. Here x_sc is the spacecraft's GSE X position (OMNI words 32; live from the ephemeris file), x_bsn is a fixed nominal bow-shock distance in `config.py` (set from the median of OMNI's bow-shock column), and vx is a 10-minute rolling median of the radial speed (OMNI word 23; live `proton_vx_gse`). If the live ephemeris is more than 6 hours old, or a source has no position, x_sc falls back to the nominal L1 distance in `config.py` and an incident is logged.

Live `time_tag` bin conventions (start or centre of the minute) are confirmed in week 1 (gate G3) and handled in the resampling function.

### 3.6 Point-in-time availability

A record is available at *t* if `observed_at` + *L* ≤ *t*, where *L* is the pipeline latency. *L* is measured live (fetch time − `time_tag`) and its 95th percentile is applied in training. Until it is measured, *L* = 5 minutes.

Index values are available at interval end + publication delay. Delays are measured live from first appearance in the JSON API (start values: 30 minutes for Hp30, 60 minutes for Kp) and applied in training.

**Known skew in index features.** Training reads the archive's revised and definitive values; live, the system sees nowcasts that GFZ revises in place. The products job archives every value as first published, and once enough has accumulated the skew report compares as-published with final values. Until then this is a documented limitation.

Available records may have `arrives_at` later than *t*. That plasma still in transit is the model's lead time.

### 3.7 Features (v1)

All features come from `features.py`, shared by training, batch serving and the online endpoint. Feature-set version is a constant; the model records it; loading fails closed on mismatch.

- **Inputs:** speed, radial velocity, field magnitude Bt, By and Bz (GSM).
- **Coupling:** v·Bs with Bs = max(−Bz, 0); the Newell coupling function v^{4/3} B_T^{2/3} sin^{8/3}(θ/2), where B_T = √(By² + Bz²) is the transverse field (GSM), not the total field `bt`, and θ = atan2(By, Bz) is the clock angle. Both are density-free.
- **In transit:** statistics of available plasma whose `arrives_at` falls 0–30, 30–60 and 60–90 minutes after *t*.
- **Recently arrived:** trailing windows of 10, 30, 60 and 180 minutes by `arrives_at`.
- **Statistics:** mean, minimum, maximum, last value and slope per window.
- **Context:** latest published Hp30 and Kp, hours since the last storm event ended, hour of day, and day of year (encoded cyclically).
- **Data health:** share of missing minutes per window, and whether the active spacecraft changed in the last 3 hours.
- **Excluded in v1:** density, temperature, dynamic pressure, and the spacecraft name itself (the model must not learn spacecraft identity).

### 3.8 Rules that must never be broken

1. **Point-in-time correctness.** Features use only what was available at *t* (§3.6).
2. **Train/serve consistency.** One propagation function, one resampling function, one feature function.
3. **Time-based splits only**, with a 15-hour embargo between training and evaluation (label horizon plus the 12-hour post-storm exclusion).
4. **At-risk rows only** in training and per-row metrics.
5. **Atomic writes.** Write to a temporary file, then rename.
6. **Raw data is immutable.** Fixes happen downstream and are recorded.

---

## 4. Evaluation

### 4.1 Periods

| Period | Dates | Use |
|---|---|---|
| Training | 1998 onward | Expanding window |
| Validation | Walk-forward folds for 2015–2022, one year each | Choose τ, calibration, hyperparameters |
| **Test** | **1 January 2023 – 30 June 2026** | Run once. Ends when SOLAR-1 became the operational source. Includes the May 2024 G5 storm, the October 2024 G4 storm and solar maximum (October 2024). |
| Live | From deployment | Accumulates; reported separately |

Walk-forward means: train on everything up to the start of the year, predict the year, step forward. The test period is predicted by four models trained to end-2022, end-2023, end-2024 and end-2025, exactly as a live system retraining annually would have done.

**Freezing.** The backtest report is generated once, committed with its inputs' manifest hash and tagged `backtest-v1`. After that the test period is treated as development data, because the promotion gate's 24-month window overlaps it. Headline numbers are never regenerated from a tuned configuration.

### 4.2 Science backtest

Inputs from OMNI (§3.4). This measures what the method can do with clean data, and is the right basis for comparing with the literature.

### 4.3 Operational backtest (headline)

The same models, the same test period, but **inputs replaced with the archived operational feeds** a live system would actually have read: DSCOVR real-time data from NCEI where available, ACE real-time data elsewhere. Run through the live ingestion code (resampling, validation, propagation, features). Archived operational records carry no fetch timestamp, so availability uses the measured live latency *L*. Where the archive carries no spacecraft position, `propagate` uses the nominal L1 distance from `config.py`; the skew report measures what that costs.

The gap between the two backtests is the cost of train/serve skew, reported as a number.

### 4.4 Skew report (`docs/skew.md`)

**Historical:** for 2016–mid-2026, the same minutes from OMNI and DSCOVR real-time, and from OMNI and ACE real-time, compared after propagating both with the shared function. Results are split by OMNI's source spacecraft, since OMNI drew mainly on Wind from 2020:

- Per variable: bias, scale ratio, correlation, and the distribution of differences, split by solar wind speed.
- Per feature: population stability index.
- Per prediction: the model's output on each input stream for the same times, and the resulting change in Brier score and hit rate.

**Ongoing:** OMNI catches up 5–6 weeks behind real time. A monthly job then compares the system's own archived SOLAR1, ACE and IMAP records with OMNI for the same minutes, so skew against the live spacecraft is measured as data accumulates. The report decides whether density or temperature enter a later feature set.

### 4.5 Metrics

| Metric | Answers |
|---|---|
| Hit rate (storms warned of) | Does it catch storms? |
| Median warning time, with interquartile range | How much notice? |
| False alarms per month; near misses; alert-hours per month | What does it cost? |
| Hit rate at NOAA's false-alarm rate | Better than NOAA when both cry wolf equally often? |
| Missed storms, listed with solar wind speed and data gaps | Why did it miss? |
| Brier score, Brier skill score vs climatology, by horizon and by time-to-onset bucket | Probability quality, and where L1 information runs out |
| Reliability diagram | When it says 20%, does a storm follow about 20% of the time? |
| Kp mean absolute error vs persistence and NOAA's 3-day forecast | Secondary output |

NOAA's 3-day forecast is compared like with like: each model forecast made at *t* is matched with NOAA's latest forecast issued before *t*.

Every metric is also re-run with SYM-H onsets (first minute SYM-H ≤ −50 nT within the event) as a sensitivity check. SYM-H falls during a storm's main phase, so these onsets are later; the check shows how much results depend on the onset definition. Events whose SYM-H never reaches −50 nT, common for storms driven by high-speed streams, are excluded from that run and counted.

### 4.6 Uncertainty

Each onset produces many correlated positive rows, so row-level intervals are far too narrow. Intervals are 90% bootstrap intervals that **resample whole storm events** for storm metrics (hit rate, warning time) and **calendar months** for false-alarm rates and Brier scores, since most Brier rows are quiet periods.

Rough scale, from GFZ Kp: about 35 events in 2024 and a faster rate in the first half of 2025, against about a dozen in 2019 near solar minimum. The test period should hold well over 100 events; the exact count comes from `labels.py`.

---

## 5. Model lifecycle

### 5.1 Model

LightGBM classifiers (one per horizon) and a LightGBM regressor for next-window Kp. Probabilities are calibrated with `CalibratedClassifierCV` wrapped around a `FrozenEstimator` (scikit-learn's replacement for `cv="prefit"`, deprecated in 1.6). Isotonic and sigmoid calibration are compared on validation folds; with few storms per fold, sigmoid wins ties.

The week-3 skeleton serves a **climatology model** through the same interfaces. The first LightGBM model replaces it via the configuration-change gate (§5.4).

### 5.2 Tracking and registry

MLflow 3.x, run as a container on the server with a SQLite backend and a local artefact store, reachable only through an SSH tunnel. Every run logs parameters, data period, manifest hash, git SHA, `uv.lock` hash, feature-set version, metrics, the model and its signature. Since 3.14 MLflow saves scikit-learn and LightGBM models with skops by default; a calibrated wrapper around LightGBM may need its types declared trusted, or the model logged as a pyfunc. A test loads every logged model in a fresh process.

The registry uses **aliases**, not the deprecated stages: `@champion` (live), `@challenger` (shadow) and `@previous` (last champion, for rollback). Serving resolves `models:/stormwarn-onset-90m@champion` at start and re-checks the alias every 60 seconds, reloading when it moves.

### 5.3 Shadow challenger

The scheduler scores `@champion` and `@challenger` on every 5-minute slot and logs both; only the champion is published. Shadow running proves a model loads, runs within latency budget and behaves sensibly on live inputs before it serves anyone. It cannot measure storm skill in a few weeks; the backtest folds do that.

### 5.4 Promotion gate

Every comparison is between models that are both **out of sample** on the data they are scored on. A champion trained before the evaluation window cannot be fairly compared with a challenger trained on part of it.

| Track | When | Evaluation | Passes if |
|---|---|---|---|
| **Routine refresh** | Same configuration, one more month of data | The refreshed model is first trained with the newest month *M* held out, so champion and refresh are both scored on *M*, which neither has seen | Trains and passes model tests; Brier score on *M* no more than 2% worse than the champion's; hit rate at fixed τ on *M* no more than 5 percentage points worse (when *M* contains storms); prediction distribution on the last 30 days within the PSI threshold of the champion's; 7 days of clean shadow running. The model is then retrained including *M*. |
| **Configuration change** | New features, hyperparameters, training data or calibration | Both configurations re-run walk-forward on identical folds: each of the latest 24 complete months predicted by a model trained on everything before it, with the embargo | Brier skill score better with bootstrap probability ≥ 0.9 (resampling months) **and** hit rate at matched false-alarm rate no more than 5 percentage points worse (resampling storms); 14 days of clean shadow running. The winning configuration is then retrained on all data. |

A single-track rule demanding P ≥ 0.9 for every refresh would almost never pass with the few dozen storms in 24 months of a declining cycle, and the model would silently freeze. Splitting the tracks keeps the model current while still demanding evidence for real changes.

Every decision, promoted or rejected, is logged in MLflow with its evidence and published on the system page.

### 5.5 Retraining schedule

Monthly, around the 20th. Training data runs to the end of the latest complete month that both sources cover: GFZ publishes definitive Kp about 12 days after month end (one observation, October 2026), but OMNI trails real time by 5–6 weeks. In practice the newest month used is usually two months back. Both lags are measured each month and logged with the run. A failed drift check can trigger an out-of-cycle refresh.

### 5.6 Rollback

`stormwarn model rollback` moves `@champion` to `@previous` and records an incident. Serving picks it up within 60 seconds. Rolling back the code is separate (§8.5).

### 5.7 Drift checks (daily, DuckDB)

- **Source changes:** every change of the active spacecraft, per data type.
- **Input drift by spacecraft:** PSI per feature against training, split by active spacecraft.
- **Performance drift:** the live signals that actually move are next-window Kp error (every 3 hours) and the quiet-period false-alarm rate. Rolling Brier score on mostly quiet rows says little and is reported but not alerted on.

---

## 6. System design

### 6.1 Architecture

```
NOAA RTSW (24 h) ─┐
NOAA alerts (30 d) ├─► archive raw ─► validate ─► resample ─► propagate ─► features ─► champion  ─► predictions ─► API
NOAA Kp products  ─┘    (immutable)    quarantine   (5 min)    (two clocks)  (shared)    challenger    (Parquet)       │
GFZ Kp + Hp30 ─────────────────────────────────────────────────────────────────────────► score ─► drift ─► gate      │
OMNI (5–6 wk lag) ─────────────────────────────────────────────────────────────────────► skew report                 │
                                                                     JSON snapshots ─► object storage ─► Astro site ◄─┘
```

### 6.2 Components

All run on one always-on server (4 vCPU, 8 GB RAM, 80 GB disk) with Docker Compose:

| Service | Role |
|---|---|
| `scheduler` | APScheduler 3.11 process running the jobs below; exposes Prometheus metrics on port 9100 |
| `api` | FastAPI; reads predictions; online `/predict` |
| `mlflow` | Tracking server and registry; bound to localhost |
| `prometheus` | Scrapes `api` and `scheduler`; remote-writes to Grafana Cloud |
| `caddy` | Reverse proxy with automatic TLS; the only public ports are 80 and 443 |

External: Grafana Cloud (free tier: 10,000 active series, 14 days' retention) for dashboards and alerting; healthchecks.io (free tier: 20 checks) as the dead-man's switch; an S3-compatible bucket for site snapshots and a separate private one for backups; GitHub Pages for the site.

APScheduler 4 is still alpha; 3.11 is used. Its default `misfire_grace_time` is one second, so every job sets it explicitly.

### 6.3 Jobs

All jobs: `max_instances=1`, `coalesce=True`, explicit `misfire_grace_time`, a healthchecks.io ping on start and on success or failure.

| Job | Schedule | Does |
|---|---|---|
| Archive | Every minute | Fetch RTSW wind, mag; write raw |
| Ingest and predict | Every 5 minutes (slot + 30 s) | Validate, resample, propagate, features, score champion and challenger, write predictions or "storm in progress" |
| Products | Every 5 minutes | Fetch alerts, NOAA Kp products, 3-day forecast, GFZ JSON (Kp, Hp30); log revisions |
| Ephemeris | Hourly | Fetch positions; staleness check |
| Score | Every 30 minutes | Join predictions to outcomes as indices arrive |
| Publish | Every 5 minutes | Write JSON snapshots to object storage |
| Compact | Daily, 00:20 UTC | Merge the previous day's small Parquet files into one |
| Drift | Daily | §5.7 |
| Re-score | Monthly | Re-score against definitive Kp; download last month's closed SWPC alerts archive file and reconcile it with the live `alerts.json` capture, logging any message the live capture missed |
| Skew | Monthly | §4.4 ongoing |
| Retrain and gate | Monthly | §5.4–5.5 |

### 6.4 Storage

Parquet on local disk; Polars for pipelines; DuckDB for scoring, metrics, drift and reports, as SQL files in `sql/`. No database server.

```
data/
├── raw/           # as received; feed=…/date=YYYY-MM-DD/; one file per fetch, compacted daily
├── quarantine/    # failed batches, with the reason
├── processed/     # validated, resampled, propagated; month=YYYY-MM/
├── predictions/   # every forecast: slot, champion and challenger outputs, model versions, feature hash
├── scores/        # predictions joined to outcomes, provisional and definitive
└── manifest.parquet  # every raw file with its SHA-256
```

### 6.5 Validation and data contracts

pandera schemas per feed (`pandera[polars]`). On every fetch:

- Normalise keys; parse timestamps as UTC; treat feed-specific sentinels as missing.
- Keep all spacecraft in `raw/`; downstream, select `active == true` per data type, then require strictly increasing, unique timestamps.
- Range checks: speed 200–3,000 km/s; |B| < 200 nT; Kp 0–9; Hp30 ≥ 0 (no upper bound).
- Gap detection: share of missing minutes per window, carried into features.
- Freshness: newest active record younger than 15 minutes, else a data-staleness incident.
- Unknown or missing keys: schema-drift incident.
- Failures quarantine the batch; the pipeline continues.

### 6.6 API

| Endpoint | Returns |
|---|---|
| `GET /health` | `ok`, `degraded` or `stale`; data age per feed; active spacecraft; last success per job; model versions |
| `GET /forecast/latest` | P₉₀, P₃ₕ and calibrated alert state, or "storm in progress"; predicted Kp; NOAA's latest active warning and forecast; model version; data age; reduced-confidence flag |
| `GET /forecast/history?hours=24` | Recent forecasts (maximum 168 hours) |
| `POST /predict` | Online inference: up to 24 hours of supplied 1-minute solar wind records plus index context in, probabilities and model version out. Runs the shared resampling, propagation and feature functions and the `@champion` model. Body size limited. |
| `GET /model` | Champion and challenger versions, training window, gate decision history |
| `GET /metrics` | Prometheus |

`/forecast/*` serves precomputed predictions, which is the system of record. `/predict` exists so that online and batch paths can be tested for parity (§8.6) and so the site's storm replay can call the live model.

### 6.7 Graceful degradation

- A failed fetch keeps the last good forecast, marked with its age; `/health` reports `degraded`.
- Missing input minutes flow into features; above 50% missing in the in-transit window, the forecast carries a reduced-confidence flag.
- If the server stops, the site keeps showing the last snapshot in object storage, and a script on the page shows "data is X hours old" from the visitor's clock.

---

## 7. Operations

### 7.1 Service level objectives

| SLO | Target | Error budget |
|---|---|---|
| **Forecast freshness:** a forecast record (prediction, "storm in progress" or explicit degraded status) is written within 2 minutes of each 5-minute slot | 99.5% of slots per calendar month | About 43 slots (3.6 hours) a month |
| **API availability:** `/health` answers 200 to an external probe | 99.5% per calendar month | About 3.6 hours a month |

**Upstream data availability** (share of slots whose active-source data is under 15 minutes old) is reported as an SLI but is not an objective, because the spacecraft are outside the system's control. The system page shows budgets remaining.

### 7.2 Metrics

- HTTP: request duration histogram, requests and errors by route.
- `stormwarn_forecast_probability{horizon}`, `stormwarn_predicted_kp`.
- `stormwarn_data_age_seconds{feed}`, `stormwarn_active_spacecraft_info{type,source}`.
- `stormwarn_job_last_success_timestamp_seconds{job}`, `stormwarn_job_duration_seconds{job}`.
- `stormwarn_forecast_slots_total{outcome="written|late|missed"}`.
- `stormwarn_quarantined_batches_total{feed}`, `stormwarn_schema_drift_total{feed}`.
- `stormwarn_feature_psi{feature,source}`, `stormwarn_rolling_kp_mae`.
- `stormwarn_model_info{alias,version}`.

### 7.3 Alerts (Grafana Cloud)

Fast and slow SLO burn rate; any job without a success in twice its interval; data older than 30 minutes; schema drift; quarantine rate above 10% in an hour; disk above 80%; PSI above 0.25 for any feature. healthchecks.io independently emails when a job stops pinging.

### 7.4 Incidents

`docs/incidents.md` from day one. Every alert that needs action gets an entry:

```markdown
## YYYY-MM-DD — Short title
- **Detected by:** (alert, check, user)
- **Impact:** (e.g. 40 minutes without forecasts; 9 slots of error budget)
- **Timeline:** (UTC)
- **Cause:**
- **Fix:**
- **Prevention:** (with link to the commit or issue)
```

### 7.5 Runbooks (`docs/runbooks.md`)

Stale data; active spacecraft switched; schema drift in a feed; model rollback; code rollback; restore from backup; disk full; endpoint retired.

### 7.6 Backups

Nightly, from the host rather than a container (a systemd timer running `deploy/backup.sh`, which pings its own healthchecks.io check): `data/predictions`, `data/scores`, `data/manifest.parquet`, raw live archives (which cannot be re-fetched), the MLflow artefacts, and the MLflow database copied consistently with `sqlite3 .backup`. Encrypted with restic to a private bucket. Retention: 7 daily, 4 weekly, 12 monthly. **A restore to a scratch directory is tested monthly** and logged; a backup never restored is not a backup.

### 7.7 Security

SSH keys only; firewall allowing 22, 80 and 443; unattended security upgrades; containers run as a non-root user; MLflow and Prometheus not exposed publicly. Secrets (deploy key, bucket credentials, healthchecks URLs, Grafana token) live in the server's `.env` and GitHub Actions secrets, never in the repository.

---

## 8. Delivery

### 8.1 Repository layout

```
stormwarn/
├── pyproject.toml · uv.lock · .python-version · README.md
├── Dockerfile · .dockerignore · compose.yaml
├── deploy/            # Caddyfile, prometheus.yml, deploy.sh, server bootstrap
├── src/stormwarn/
│   ├── cli.py
│   ├── config.py           # endpoints, thresholds, distances, latencies, schedules
│   ├── ingest/             # swpc.py, gfz.py, omni.py, operational_archive.py
│   ├── schemas.py          # pandera contracts
│   ├── validate.py
│   ├── resample.py         # 1-minute to 5-minute; shared
│   ├── propagation.py      # observed_at → arrives_at; shared
│   ├── features.py         # shared; FEATURE_SET_VERSION
│   ├── labels.py           # events, onsets, risk set, labels
│   ├── alerts.py           # alert logic, applied to the model and to NOAA
│   ├── train.py · calibrate.py · evaluate.py · bootstrap.py
│   ├── registry.py         # aliases, loading, rollback
│   ├── gate.py
│   ├── monitor.py          # scoring, drift, skew
│   ├── publish.py · backup.py · scheduler.py
│   └── api/main.py
├── sql/
├── tests/  (unit/, contract/, fixtures/ with recorded real payloads)
├── site/
├── docs/   (data.md, backtest.md, skew.md, decisions.md, runbooks.md, incidents.md)
└── .github/
    ├── dependabot.yml
    └── workflows/ (ci.yml, deploy.yml, site.yml, canary.yml)
```

### 8.2 Python and dependencies

Python 3.13, pinned in `.python-version` and `requires-python = ">=3.13,<3.14"`. uv 0.13 downloads Python 3.15 by default when nothing is pinned, so the pin matters.

```bash
uv init --package --python 3.13 stormwarn
cd stormwarn
# set requires-python = ">=3.13,<3.14" in pyproject.toml; check .python-version says 3.13
uv add polars duckdb httpx fastapi "uvicorn[standard]" lightgbm scikit-learn \
       mlflow prometheus-client "apscheduler>=3.11,<4" typer "pandera[polars]" netcdf4
uv add --dev pytest pytest-cov hypothesis ruff respx
```

Polars 2.0 was released on 6 October 2026. Whether `pandera`'s Polars backend supports it is checked in week 1 (gate G4); if not, pin `polars>=1,<2` until it does.

CLI (Typer, under `[project.scripts]`):

```bash
uv run stormwarn backfill          # OMNI 1-minute, GFZ, ACE and DSCOVR operational archives, NCEI forecasts
uv run stormwarn labels
uv run stormwarn train --horizon 90m
uv run stormwarn backtest --inputs science|operational
uv run stormwarn skew-report
uv run stormwarn gate --track refresh|config
uv run stormwarn model rollback
uv run stormwarn scheduler
```

### 8.3 Dockerfile

Based on uv's Docker guide.

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# LightGBM's Linux wheels need the OpenMP runtime, which slim images lack
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN useradd --uid 1000 --create-home app
USER app
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "stormwarn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`.dockerignore` excludes `data/`, `mlflow/`, `.venv/` and `site/`. Pin the base image by digest once chosen.

### 8.4 Compose

```yaml
x-app: &app
  image: ghcr.io/${GH_OWNER}/stormwarn:${IMAGE_TAG:?set IMAGE_TAG}
  env_file: .env
  restart: unless-stopped
  volumes: ["./data:/app/data"]

services:
  api:
    <<: *app
    ports: ["127.0.0.1:8000:8000"]   # public traffic goes through Caddy
    depends_on:
      mlflow: { condition: service_healthy }
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s

  scheduler:
    <<: *app
    command: ["stormwarn", "scheduler"]
    depends_on:
      mlflow: { condition: service_healthy }

  mlflow:
    image: ghcr.io/mlflow/mlflow:${MLFLOW_VERSION:?}
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/artifacts
    volumes: ["./mlflow:/mlflow"]
    ports: ["127.0.0.1:5000:5000"]
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 30s
      retries: 5
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:${PROMETHEUS_VERSION:?}
    volumes: ["./deploy/prometheus.yml:/etc/prometheus/prometheus.yml:ro", "./prometheus:/prometheus"]
    restart: unless-stopped

  caddy:
    image: caddy:2
    ports: ["80:80", "443:443"]
    volumes: ["./deploy/Caddyfile:/etc/caddy/Caddyfile:ro", "./caddy:/data"]
    depends_on:
      api: { condition: service_healthy }
    restart: unless-stopped
```

`.env` sets `GH_OWNER`, `IMAGE_TAG`, `MLFLOW_VERSION` and `PROMETHEUS_VERSION`. Use each registry's tag format exactly (Prometheus tags carry a `v` prefix, e.g. `v3.15.0`; check MLflow's on GHCR) and keep MLflow's matching the version in `uv.lock`. The server bootstrap script makes `./data` writable by UID 1000 (the app user) and `./prometheus` by UID 65534 (Prometheus runs as `nobody`). MLflow's server has added host-header and CORS protections since 3.5; if the scheduler's requests to `http://mlflow:5000` are refused, allow that host name using the option documented for the pinned version. MLflow 3.17 also requires a database schema upgrade with all writers stopped when upgrading from earlier 3.x versions.

### 8.5 CI/CD

**`ci.yml`** on every push and pull request: `uv sync --locked`, `ruff check`, `ruff format --check`, `pytest -m "not live" --cov`. Build the Docker image to prove it builds.

**`deploy.yml`** on push to `main` after CI passes:

1. Build and push `ghcr.io/<owner>/stormwarn:<git-sha>` (permissions `contents: read`, `packages: write`).
2. Over SSH: `deploy/deploy.sh <git-sha>`, which records the running tag, sets `IMAGE_TAG`, runs `docker compose pull` and `docker compose up -d`, then polls `/health`.
3. If `/health` does not return 200 with the new version and a loaded model within 3 minutes, `deploy.sh` re-deploys the recorded tag and the workflow fails. (`/health` returns 200 whenever the process works; stale upstream data shows in its body as `stale`, so an upstream outage never triggers a rollback.)

Manual rollback: `deploy/deploy.sh <previous-sha>`, documented in the runbook.

**`canary.yml`** daily: hits every endpoint in `config.py` and validates the response against its contract. A failure opens an issue. This catches retired or reshaped endpoints before the pipeline does.

**`site.yml`** on changes under `site/`: build Astro with `withastro/action` and deploy with `actions/deploy-pages`.

Actions are pinned to full commit SHAs, with Dependabot keeping them current. `astral-sh/setup-uv` no longer publishes floating major tags (current: v10.3.0), so a SHA or exact version is required in any case. A monthly scheduled workflow runs `uv lock --upgrade` and opens a pull request.

### 8.6 Tests

| Type | Examples |
|---|---|
| Unit | Events under 12 hours apart merge; onset refinement and fallback; risk set starts at onset when onset falls in *W*₋₁ |
| Property (Hypothesis) | Alert logic: debounce never splits a run shorter than 30 minutes; warning time never exceeds the horizon; every alert is a hit or a false alarm or still open |
| Parsers | Recorded real payloads, including May 2024 GFZ data, the SWPC archive file `alerts_202405.html` (294 messages; the storm's K5 warning chain, serials 1874 → 1875 → 1876 → 1878 → 1879 (1877 is not part of it), must resolve to one warning valid 10 May 17:00 – 13 May 21:00 UTC), October 2026 alerts with label variants, and the 12 October 2026 three-spacecraft RTSW batch |
| Contract | pandera schemas accept recorded payloads and reject mutated ones |
| Propagation | An OMNI record is unavailable before `observed_at` + *L* and available before `arrives_at`; OMNI and live records with the same physics get the same `arrives_at` |
| **Leakage** | Features at *t* are identical whether computed with or without data available after *t* |
| **Parity** | Training, batch serving and `POST /predict` produce identical features and predictions for the same input |
| Model | Beats climatology on Brier score on a fixed validation slice; probabilities in [0, 1]; loading a model with the wrong feature-set version fails closed |
| Gate | A degraded challenger is rejected; a copy of the champion is not promoted on the configuration track; a refresh passes |
| API | Schemas; stale data reported as stale; storm in progress reported; body limit enforced |
| Live (`-m live`, canary only) | Every endpoint responds and matches its contract |

### 8.7 Versions at time of writing (13 October 2026)

uv 0.13.0 · MLflow 3.17.0 · APScheduler 3.11.3 · LightGBM 4.7.0 · Polars 2.0.0 · DuckDB 1.5.6 · scikit-learn 1.9.1 · FastAPI 0.143.0 · prometheus-client 0.26.0 · pandera 0.34.1 · Ruff 0.17.0 · pytest 9.1.1 · Prometheus 3.15 (LTS 3.13) · Astro 7.

Exact versions live in `uv.lock` and `.env`; this list only records the starting point.

---

## 9. Website

An Astro site on GitHub Pages that reads JSON snapshots from public object storage, so it rebuilds only when its code changes and never commits data to the repository.

- **Now:** P₉₀ or "storm in progress", alert state, NOAA's latest warning, data age, staleness notice.
- **Results:** headline table from the operational backtest with intervals; science backtest alongside; skew cost.
- **Trigger slider:** hit rate, warning time, false alarms and alert-hours for τ in steps of 0.05, precomputed.
- **Storm replay:** May 2024, October 2024 and October 2026: solar wind in transit, probability rising, model alert vs NOAA warning, onset.
- **Calibration** and **Kp vs persistence and NOAA**.
- **System:** SLOs and error budgets, data age, active spacecraft, job status, drift, gate decisions, incidents.

---

## 10. Plan

Sixteen weeks of evenings. Something real runs on the server from week 1.

| Weeks | Deliverables | Done when |
|---|---|---|
| **1** | Server provisioned and hardened; archive and products jobs live (RTSW, ephemeris, alerts, NOAA Kp products, 3-day forecast, GFZ JSON) writing raw Parquet; healthchecks.io; nightly backup; week-1 gates G2–G4 | Raw archives have run unattended for 72 hours; gates resolved and recorded in `docs/decisions.md` |
| **2–3** | **Walking skeleton:** validation and quarantine; resample, propagate, features (first cut); climatology model registered as `@champion`; predictions every 5 minutes; API; CI, image to GHCR, automated deploy with rollback; Prometheus, Grafana Cloud, SLO dashboard and alerts | A commit to `main` reaches production unattended; a deliberately broken deploy rolls itself back; `/forecast/latest` is live |
| **4–6** | Backfill: OMNI 1-minute 1998–2026, GFZ, ACE and DSCOVR operational archives, NCEI forecasts, SWPC alerts archive 2002–2026 (with a parser for warnings, extensions, cancellations, alerts and shock reports). Labels, onsets (Hp30, NOAA-aligned, SYM-H), risk set. Full feature set. Leakage and parity tests | `docs/data.md` published, including event counts per year and the spacecraft findings |
| **7–9** | LightGBM, walk-forward validation, calibration, τ; skew report; science and operational backtests on the test period, run once and tagged `backtest-v1`; shadow scoring in the scheduler; LightGBM registered as `@challenger` and running in shadow | `docs/backtest.md` and `docs/skew.md` published: **the headline numbers** |
| **10–12** | First configuration-change gate run promotes LightGBM over climatology after 14 days in shadow; live scoring and re-scoring; drift checks; monthly retrain with both gate tracks; `POST /predict`; runbooks; first restore drill | LightGBM is champion through the gate; a monthly retrain has run and its decision is logged and published |
| **13–16** | Website; replays; README results; incident review; tidy | Public links work; README explains every decision |

---

## 11. Decisions log

| # | Decision | Reason |
|---|---|---|
| 1 | Space weather as the domain | Ready-made ground truth, a physical lead time, small data, real drift |
| 2 | Storms are Kp ≥ 4.66 (5-), matching SWPC's G1 labelling | Enough events to train and evaluate; matches NOAA's scale |
| 3 | Events from Kp, onsets timed with Hp30 | Kp's 3-hour windows are coarser than the lead time |
| 4 | Risk set excludes onset to event end + 12 h | Onsets cannot recur then; starting at onset avoids mislabelled storm rows |
| 5 | Primary horizon 90 minutes; 3 hours secondary | Matches the physical L1 lead time; 3 hours links to the literature |
| 6 | Alerts debounced; warning time capped at the horizon; same rules for NOAA | Stops flicker and open-ended credit; keeps the comparison fair |
| 7 | Walking skeleton with a climatology model by week 3 | Engineering risk is retired first; every later model passes the same gate |
| 8 | Archive live feeds from week 1 | RTSW keeps 24 hours and alerts 30 days; missed data cannot be recovered |
| 9 | Train on OMNI 1-minute, filtered by spacecraft position | Clean, long record; position is a better L1 filter than ID |
| 10 | Headline from an operational backtest on archived real-time inputs | Live inputs come from different spacecraft and processing than training |
| 11 | No density or temperature features in v1 | Live spacecraft disagree on both by factors of 3 to more than 10; the skew report decides |
| 12 | One propagation function using spacecraft position and radial speed | OMNI's shift only recovers observation time; archive and live are propagated identically |
| 13 | Test period 2023 to 30 June 2026, frozen after one run | Ends at the switch to SOLAR-1; freezing protects headline numbers from gate reuse |
| 14 | Gate compares training procedures on identical folds, in two tracks | Artefact comparison is contaminated; a single strict track would freeze the model |
| 15 | MLflow aliases, shadow challenger, one-command rollback | Stages are deprecated; shadowing proves operability before promotion |
| 16 | GHCR images by commit SHA, automated deploy with health-checked rollback | Every deploy is reproducible and reversible |
| 17 | Two SLOs; upstream data availability as an SLI only | Objectives should cover what the system controls |
| 18 | Prometheus with Grafana Cloud, healthchecks.io as dead-man's switch | Metrics that nothing reads are decoration |
| 19 | Parquet with Polars, SQL in DuckDB, no database server | Small data; less to operate |
| 20 | Site reads snapshots from object storage | No commit churn; survives server outages |
| 21 | One always-on server, no Kubernetes | Simplest thing that works; no cold starts |
| 22 | Bootstrap by storm event and by month | Rows within a storm are not independent |
| 23 | Wing Kp and Geospace predicted Kp not used | Discontinued in 2018; frozen since June 2024 |
| 24 | Model-versus-NOAA claims must hold under both Hp30 and NOAA-aligned onsets | NOAA forecasts its own K-index; in May 2024 the Hp30 onset preceded NOAA's K = 5 by nearly three hours |
| 25 | NOAA's historical warnings from SWPC's FTP archive, reconciled monthly with the live capture | Verified 2002–2026 coverage; reconciliation measures the live capture's completeness |

---

## 12. Known limitations

- **Lead time is bounded by physics**, and fast eruptions give the least notice.
- **NOAA warns from the same data**, and its forecasters can also use solar observations.
- **Onset timing is approximate and definition-dependent.** Hp30 resolves to 30 minutes, has no definitive version and may be revised. It can reach storm level hours before NOAA's K-index does (§2.7), so results are reported under Hp30, NOAA-aligned and SYM-H onsets.
- **Propagation is simplified.** A flat delay ignores the tilt of solar wind structures; errors of several minutes are normal and some events are off by much more.
- **The live spacecraft are new.** SOLAR-1 has been operational only since 30 June 2026 and has no published validation against Wind or ACE yet. The skew report measures this as data accumulate.
- **Storms are becoming rarer.** Solar Cycle 25 peaked in October 2024; the next minimum is expected around 2030. Live storm counts will be small and the backtest carries the headline.
- **Live indices are nowcasts**; scores are revised when definitive values arrive.

---

## 13. Week-1 gates and open questions

| Gate | Question | If it fails |
|---|---|---|
| **G1** | ✅ **Passed 13 October 2026.** SWPC's alerts archive holds K-index warnings with issue times, validity windows and extensions from 2002 to the current month. | — |
| **G2** | Real latency of each feed, measured from the server for 72 hours | Sets *L* and index publication delays (§3.6) |
| **G3** | Is `rtsw_ephemerides_1h.json` current? What bin convention does `time_tag` use? | Fall back to a fixed L1 distance for x_sc; document the convention |
| **G4** | Does pandera's Polars backend support Polars 2? | Pin Polars 1.x |

Open, not blocking:

- How far NCEI's DSCOVR real-time archive extends after June 2026 (irrelevant for the test period, which ends then).
- Where IMAP real-time data can be downloaded historically, for the skew report.
- Server provider and monthly budget; bucket provider for snapshots and backups.

---

## 14. References

- Wing, S. et al. (2005). Kp forecast models. *JGR Space Physics* 110. doi:10.1029/2004JA010500
- Chakraborty, S. and Morley, S. K. (2020). Probabilistic prediction of geomagnetic storms and the Kp index. *J. Space Weather Space Clim.* 10, 36. doi:10.1051/swsc/2020037
- Zhelavskaya, I. S. et al. (2019). Systematic analysis of machine learning and feature selection techniques for prediction of the Kp index. *Space Weather* 17(10), 1461–1486. doi:10.1029/2019SW002271
- Camporeale, E. (2019). The challenge of machine learning in space weather: Nowcasting and forecasting. *Space Weather*. doi:10.1029/2018SW002061
- Liemohn, M. W. et al. (2018). Model evaluation guidelines for geomagnetic index predictions. *Space Weather* 16(12), 2079–2102. doi:10.1029/2018SW002067
- Smith, A. W. et al. (2020). Probabilistic forecasts of storm sudden commencements from interplanetary shocks using machine learning. *Space Weather* 18(11), e2020SW002603. doi:10.1029/2020SW002603 (Brier skill, ROC and reliability for L1-driven forecasts.)
- Matzka, J. et al. (2021). The geomagnetic Kp index and derived indices of geomagnetic activity. *Space Weather* 19(5), e2020SW002641. doi:10.1029/2020SW002641
- Matzka, J. et al. (2021). Geomagnetic Kp index. V. 1.0. GFZ Data Services. doi:10.5880/Kp.0001
- Matzka, J. et al. (2024). Geomagnetic Hpo index. V. 3.0. GFZ Data Services. doi:10.5880/Hpo.0003
- Yamazaki, Y. et al. (2022). Geomagnetic activity index Hpo. *Geophysical Research Letters* 49, e2022GL098860. doi:10.1029/2022GL098860
- Yamazaki, Y. et al. (2024). Assessment of geomagnetic activity for the Kp=9 "Gannon Storm" in May 2024 based on version 3.0 Hpo indices. ESS Open Archive preprint. doi:10.22541/essoar.171838396.68563140/v1 (cited in the current Hpo file headers)
- Gonzalez, W. D. et al. (1994). What is a geomagnetic storm? *JGR* 99. doi:10.1029/93JA02867
- Loto'aniu, T. M. et al. (2022). The DSCOVR solar wind mission. *Space Weather* 20, e2022SW003085.
- King, J. H. and Papitashvili, N. E. (2005). Solar wind spatial scales in and comparisons of hourly Wind and ACE plasma and magnetic field data. *JGR* 110. doi:10.1029/2004JA010649 (OMNI methodology.)

Titles and DOIs were checked against publisher, repository or GFZ pages in October 2026. Article numbers for Wing et al. and Camporeale were not confirmed; add them before publishing the backtest report.

---

## 15. Glossary

| Term | Plain English |
|---|---|
| L1 | A point about 1.5 million km sunward where monitoring spacecraft sit |
| Bow shock | Where the solar wind first meets Earth's magnetic field |
| Bz | North–south component of the solar wind's magnetic field; southward Bz drives storms |
| Kp / Hp30 | Indices of geomagnetic disturbance over 3 hours / 30 minutes |
| SYM-H | One-minute index of the ring current, which strengthens during storms |
| G-scale | NOAA's storm scale, G1 (minor) to G5 (extreme) |
| Onset | The moment a storm begins |
| Risk set | The moments when a new storm could begin |
| Science vs operational data | Reprocessed archive data vs what was available in real time |
| Train/serve skew | Differences between the data a model learned from and the data it receives live |
| Walking skeleton | The thinnest end-to-end version of a system, deployed before its parts are good |
| Champion / challenger | The live model, and a candidate running in shadow |
| Shadow mode | Running a model on live inputs without publishing its outputs |
| SLO / SLI / error budget | A reliability target / the measurement behind it / the failure allowed before the target is missed |
| Brier skill score | Improvement in squared probability error over a baseline; above zero is better |
| Bootstrap | Estimating uncertainty by resampling, here whole storms at a time |
| PSI | Population stability index, a measure of distribution shift |
| Dead-man's switch | An alarm that fires when an expected "I'm alive" signal stops |
| Point-in-time correctness | Using only information that existed at the moment of prediction |
