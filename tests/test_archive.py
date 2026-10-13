import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest
from typer.testing import CliRunner

from stormwarn import cli
from stormwarn.config import FEEDS, GFZ_JSON
from stormwarn.ingest import archive
from stormwarn.ingest.archive import make_client, run_archive, to_records

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "feeds.json").read_text())
NOW = datetime(2026, 10, 13, 14, 7, tzinfo=UTC)


def no_sleep(_: float) -> None:
    pass


class FakeUpstream:
    """Serves fixture payloads by URL; individual feeds can be made to fail."""

    def __init__(self, payloads: dict | None = None) -> None:
        self.payloads = copy.deepcopy(payloads or FIXTURES)
        self.status: dict[str, int] = {}
        self.calls: dict[str, int] = {}

    def feed_for(self, request: httpx.Request) -> str:
        url = str(request.url)
        if url.startswith(GFZ_JSON):
            return {"Kp": "gfz_kp", "Hp30": "gfz_hp30"}[request.url.params["index"]]
        return next(f.name for f in FEEDS if f.url == url)

    def handler(self, request: httpx.Request) -> httpx.Response:
        feed = self.feed_for(request)
        self.calls[feed] = self.calls.get(feed, 0) + 1
        status = self.status.get(feed, 200)
        if status != 200:
            return httpx.Response(status)
        return httpx.Response(200, json=self.payloads[feed])

    def client(self) -> httpx.Client:
        return make_client(httpx.MockTransport(self.handler))


def read_feed(root: Path, feed: str) -> pl.DataFrame:
    return pl.read_parquet(root / f"feed={feed}", hive_partitioning=False)


def test_to_records_handles_objects_header_rows_and_parallel_arrays():
    assert to_records([{"a": 1}]) == [{"a": 1}]
    assert to_records([["time_tag", "Kp"], ["t1", 1.0], ["t2", 2.0]]) == [
        {"time_tag": "t1", "Kp": 1.0},
        {"time_tag": "t2", "Kp": 2.0},
    ]
    records = to_records(FIXTURES["gfz_kp"])
    assert records[0] == {"datetime": "2026-10-12T15:00:00Z", "Kp": 1.0, "status": "pre"}
    assert len(records) == 3


def test_to_records_rejects_unknown_shapes():
    with pytest.raises(ValueError):
        to_records({"error": "no data"})


def test_first_run_archives_every_record(tmp_path):
    results = run_archive(tmp_path, FakeUpstream().client(), now=NOW, sleep=no_sleep)

    assert all(r.ok for r in results)
    for r in results:
        assert r.n_new == r.n_records > 0
    wind = read_feed(tmp_path, "rtsw_wind_1m")
    assert set(wind["source"]) == {"IMAP", "SOLAR1", "ACE"}
    assert wind.schema["first_seen_at"] == pl.Datetime("us", "UTC")
    expected = tmp_path / "feed=rtsw_wind_1m" / "date=2026-10-13" / "20261013T140700Z.parquet"
    assert expected.exists()


def test_second_identical_run_stores_nothing_new(tmp_path):
    upstream = FakeUpstream()
    run_archive(tmp_path, upstream.client(), now=NOW, sleep=no_sleep)
    results = run_archive(tmp_path, upstream.client(), now=NOW.replace(hour=15), sleep=no_sleep)

    assert all(r.n_new == 0 for r in results)
    assert len(list((tmp_path / "feed=rtsw_wind_1m").rglob("*.parquet"))) == 1
    assert len(list((tmp_path / "_fetches").rglob("*.parquet"))) == 2


def test_revised_value_is_kept_alongside_original(tmp_path):
    upstream = FakeUpstream()
    run_archive(tmp_path, upstream.client(), now=NOW, sleep=no_sleep)
    upstream.payloads["gfz_kp"]["Kp"][2] = 1.333  # GFZ revises a nowcast in place
    results = run_archive(tmp_path, upstream.client(), now=NOW.replace(hour=15), sleep=no_sleep)

    by_feed = {r.feed: r for r in results}
    assert by_feed["gfz_kp"].n_new == 1
    kp = read_feed(tmp_path, "gfz_kp").filter(pl.col("record_time") == "2026-10-12T21:00:00Z")
    values = sorted(json.loads(s)["Kp"] for s in kp["record_json"])
    assert values == [1.0, 1.333]


def test_failing_feed_is_retried_and_does_not_stop_others(tmp_path):
    upstream = FakeUpstream()
    upstream.status["rtsw_mag_1m"] = 503
    results = run_archive(tmp_path, upstream.client(), now=NOW, sleep=no_sleep)

    by_feed = {r.feed: r for r in results}
    assert by_feed["rtsw_mag_1m"].error == "HTTP 503"
    assert upstream.calls["rtsw_mag_1m"] == archive.ATTEMPTS
    assert by_feed["rtsw_wind_1m"].ok and by_feed["rtsw_wind_1m"].n_new == 3

    log = pl.read_parquet(tmp_path / "_fetches", hive_partitioning=False)
    assert log.filter(pl.col("feed") == "rtsw_mag_1m")["http_status"].to_list() == [503]


def test_client_errors_are_not_retried(tmp_path):
    upstream = FakeUpstream()
    upstream.status["noaa_kp"] = 404
    run_archive(tmp_path, upstream.client(), now=NOW, sleep=no_sleep)
    assert upstream.calls["noaa_kp"] == 1


def test_no_temporary_files_are_left(tmp_path):
    (tmp_path / "feed=x").mkdir()
    (tmp_path / "feed=x" / ".leftover.parquet.tmp").write_text("partial")
    run_archive(tmp_path, FakeUpstream().client(), now=NOW, sleep=no_sleep)
    assert not list(tmp_path.rglob("*.tmp"))


def test_cli_exits_2_when_a_feed_fails(tmp_path, monkeypatch):
    upstream = FakeUpstream()
    upstream.status["swpc_alerts"] = 500
    original = archive.run_archive
    monkeypatch.setattr(
        cli,
        "run_archive",
        lambda root, feeds: original(root, upstream.client(), feeds, NOW, no_sleep),
    )

    result = CliRunner().invoke(cli.app, ["archive", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "HTTP 500" in result.output
