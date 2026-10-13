# Decisions

The full decision table is in [`brief.md`](brief.md#11-decisions-log). This file records decisions made during the build, with the evidence behind them.

## 2026-10-13: Archive live feeds from GitHub Actions until the server exists

**Context.** NOAA keeps real-time solar wind for 24 hours and alerts for about 30 days; GFZ revises nowcasts in place. Anything not captured is lost. The server is weeks away and the project budget is £0.

**Decision.** An hourly scheduled GitHub Actions workflow fetches nine feeds and commits new records to a separate public repository, `stormwarn-data`.

**Why it is safe.** Every feed reaches back at least 24 hours, so hourly polling loses nothing even if GitHub delays or skips several runs. healthchecks.io alerts if runs stop. A separate repository keeps the code repository small to clone.

**Storage format.**
- One Parquet file per feed per run, holding only records not stored before. A record is new if its exact content is new, so in-place revisions (GFZ nowcasts, NOAA estimates) are kept as separate rows with their own `first_seen_at`.
- Records are stored as canonical JSON strings, not typed columns. The raw layer must never fail because upstream changed a field; typing and validation happen downstream.
- Floats are re-serialised by Python's JSON encoder (`4.00` becomes `4.0`). Values are unchanged; byte-for-byte fidelity is not kept. Each response's SHA-256 is logged per fetch.
- If the duplicate check misses (lookback too short), a duplicate is stored. Duplicates are harmless downstream; gaps are not.

**Limits.** `first_seen_at` is only accurate to the polling interval, so feed latency (gate G2) is measured on the server, not here.
