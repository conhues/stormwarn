import os
from pathlib import Path
from typing import Annotated

import typer

from stormwarn.config import FEEDS
from stormwarn.ingest.archive import FetchResult, run_archive

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Storm Warning: geomagnetic storm onset forecasting."""


def _summary(results: list[FetchResult]) -> str:
    lines = [
        "| Feed | Status | Records | New | Newest record | Seconds |",
        "|---|---|---:|---:|---|---:|",
    ]
    for r in results:
        status = "ok" if r.ok else f"**{r.error}**"
        lines.append(
            f"| {r.feed} | {status} | {r.n_records} | {r.n_new} "
            f"| {r.newest_record_time or ''} | {r.duration_s} |"
        )
    return "\n".join(lines)


@app.command()
def archive(
    root: Annotated[Path, typer.Option(help="Archive directory.")] = Path("data/archive"),
    feed: Annotated[
        list[str] | None, typer.Option(help="Only these feeds (repeatable). Default: all.")
    ] = None,
) -> None:
    """Fetch every upstream feed and store records not seen before.

    Exits 2 if any feed failed; the others are still archived.
    """
    feeds = [f for f in FEEDS if not feed or f.name in feed]
    unknown = set(feed or []) - {f.name for f in FEEDS}
    if unknown:
        raise typer.BadParameter(f"unknown feed(s): {', '.join(sorted(unknown))}")

    results = run_archive(root, feeds=feeds)
    summary = _summary(results)
    typer.echo(summary)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a") as fh:
            fh.write(f"### Archive run\n\n{summary}\n")
    if not all(r.ok for r in results):
        raise typer.Exit(code=2)
