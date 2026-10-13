"""Archive upstream feeds that are only kept for a short time.

Each run fetches every feed and stores the records it has not seen before, one Parquet
file per feed per run:

    <root>/feed=<name>/date=<YYYY-MM-DD>/<fetch_id>.parquet
    <root>/_fetches/date=<YYYY-MM-DD>/<fetch_id>.parquet   (one row per feed per run)

Records are stored untyped, as canonical JSON, so an upstream schema change can never
break archiving. Typing and validation happen downstream. A record is "new" if its exact
content has not been stored before, so revisions of the same timestamp are kept too.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from stormwarn.config import FEEDS, USER_AGENT, Feed

RECORD_SCHEMA = {
    "record_time": pl.String,
    "source": pl.String,
    "record_json": pl.String,
    "record_sha256": pl.String,
    "first_seen_at": pl.Datetime("us", "UTC"),
    "fetch_id": pl.String,
}

FETCH_SCHEMA = {
    "fetch_id": pl.String,
    "feed": pl.String,
    "url": pl.String,
    "started_at": pl.Datetime("us", "UTC"),
    "duration_s": pl.Float64,
    "http_status": pl.Int32,
    "response_bytes": pl.Int64,
    "response_sha256": pl.String,
    "n_records": pl.Int64,
    "n_new": pl.Int64,
    "newest_record_time": pl.String,
    "error": pl.String,
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
ATTEMPTS = 3


@dataclass
class FetchResult:
    feed: str
    url: str
    started_at: datetime
    duration_s: float = 0.0
    http_status: int | None = None
    response_bytes: int | None = None
    response_sha256: str | None = None
    n_records: int = 0
    n_new: int = 0
    newest_record_time: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        transport=transport,
    )


def to_records(payload: Any) -> list[dict[str, Any]]:
    """Normalise the three payload shapes seen upstream into a list of dicts."""
    # GFZ JSON API: parallel arrays keyed by name, e.g. {"datetime": [...], "Kp": [...]}.
    if isinstance(payload, dict) and isinstance(payload.get("datetime"), list):
        n = len(payload["datetime"])
        columns = {k: v for k, v in payload.items() if isinstance(v, list) and len(v) == n}
        return [{k: columns[k][i] for k in columns} for i in range(n)]
    if isinstance(payload, list):
        # Older SWPC products: an array of arrays whose first row is the header.
        if payload and all(isinstance(row, list) for row in payload):
            header, *rows = payload
            return [dict(zip(header, row, strict=True)) for row in rows]
        if all(isinstance(row, dict) for row in payload):
            return payload
    raise ValueError(f"unexpected payload shape: {type(payload).__name__}")


def canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def get_with_retries(
    client: httpx.Client,
    url: str,
    params: dict[str, str] | None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if attempt == ATTEMPTS:
                raise
        else:
            if response.status_code not in RETRYABLE_STATUS or attempt == ATTEMPTS:
                response.raise_for_status()
                return response
        sleep(2.0**attempt)
    raise AssertionError("unreachable")


def partition_dir(root: Path, feed: str, day: date) -> Path:
    return root / f"feed={feed}" / f"date={day.isoformat()}"


def seen_hashes(root: Path, feed: Feed, today: date) -> set[str]:
    files = [
        f
        for days_back in range(feed.lookback_days + 1)
        for f in partition_dir(root, feed.name, today - timedelta(days=days_back)).glob("*.parquet")
    ]
    if not files:
        return set()
    hashes = pl.read_parquet(files, columns=["record_sha256"], hive_partitioning=False)
    return set(hashes["record_sha256"].to_list())


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write to a temporary file, then rename, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.write_parquet(tmp, compression="zstd")
    os.replace(tmp, path)


def new_records_frame(
    records: Iterable[dict[str, Any]],
    feed: Feed,
    seen: set[str],
    first_seen_at: datetime,
    fetch_id: str,
) -> pl.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for record in records:
        body = canonical_json(record)
        digest = sha256(body)
        if digest in seen or digest in rows:
            continue
        time_value = record.get(feed.time_key)
        source = record.get("source")
        rows[digest] = {
            "record_time": None if time_value is None else str(time_value),
            "source": None if source is None else str(source),
            "record_json": body,
            "record_sha256": digest,
            "first_seen_at": first_seen_at,
            "fetch_id": fetch_id,
        }
    return pl.DataFrame(list(rows.values()), schema=RECORD_SCHEMA)


def archive_feed(
    client: httpx.Client,
    feed: Feed,
    root: Path,
    now: datetime,
    fetch_id: str,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchResult:
    params = feed.params(now) if feed.params else None
    result = FetchResult(feed=feed.name, url=feed.url, started_at=datetime.now(UTC))
    t0 = time.monotonic()
    try:
        response = get_with_retries(client, feed.url, params, sleep)
        result.url = str(response.url)
        result.http_status = response.status_code
        result.response_bytes = len(response.content)
        result.response_sha256 = sha256(response.content)

        records = to_records(response.json())
        result.n_records = len(records)
        times = [str(r[feed.time_key]) for r in records if r.get(feed.time_key) is not None]
        result.newest_record_time = max(times) if times else None

        today = now.date()
        new = new_records_frame(
            records, feed, seen_hashes(root, feed, today), datetime.now(UTC), fetch_id
        )
        result.n_new = new.height
        if new.height:
            path = partition_dir(root, feed.name, today) / f"{fetch_id}.parquet"
            write_parquet_atomic(new, path)
    except httpx.HTTPStatusError as exc:
        result.http_status = exc.response.status_code
        result.error = f"HTTP {exc.response.status_code}"
    except Exception as exc:  # one bad feed must never stop the others
        result.error = f"{type(exc).__name__}: {exc}"[:500]
    result.duration_s = round(time.monotonic() - t0, 3)
    return result


def run_archive(
    root: Path,
    client: httpx.Client | None = None,
    feeds: Iterable[Feed] = FEEDS,
    now: datetime | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[FetchResult]:
    now = now or datetime.now(UTC)
    fetch_id = now.strftime("%Y%m%dT%H%M%SZ")
    for stale in root.rglob(".*.tmp"):  # left behind by a run killed mid-write
        stale.unlink()

    owns_client = client is None
    client = client or make_client()
    try:
        results = [archive_feed(client, feed, root, now, fetch_id, sleep) for feed in feeds]
    finally:
        if owns_client:
            client.close()

    log = pl.DataFrame([{"fetch_id": fetch_id, **vars(r)} for r in results], schema=FETCH_SCHEMA)
    write_parquet_atomic(
        log, root / "_fetches" / f"date={now.date().isoformat()}" / f"{fetch_id}.parquet"
    )
    return results
