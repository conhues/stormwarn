"""Every upstream endpoint lives here, so a retired or moved feed is a one-line change."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

SWPC = "https://services.swpc.noaa.gov"
GFZ_JSON = "https://kp.gfz.de/app/json/"
USER_AGENT = "stormwarn/0.1 (+https://github.com/conhues/stormwarn)"


def _gfz_params(index: str) -> Callable[[datetime], dict[str, str]]:
    def params(now: datetime) -> dict[str, str]:
        end = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start = end - timedelta(days=3)
        return {
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "index": index,
        }

    return params


@dataclass(frozen=True)
class Feed:
    name: str
    url: str
    time_key: str
    # How many days of earlier archive partitions to check for records already stored.
    # Must exceed how far back the upstream file reaches. Too short only stores duplicates;
    # too long only costs read time. Neither loses data.
    lookback_days: int
    params: Callable[[datetime], dict[str, str]] | None = None


FEEDS: tuple[Feed, ...] = (
    # Real-time solar wind: SWPC keeps the last 24 hours, every spacecraft interleaved.
    Feed("rtsw_wind_1m", f"{SWPC}/json/rtsw/rtsw_wind_1m.json", "time_tag", 2),
    Feed("rtsw_mag_1m", f"{SWPC}/json/rtsw/rtsw_mag_1m.json", "time_tag", 2),
    Feed("rtsw_ephemerides_1h", f"{SWPC}/json/rtsw/rtsw_ephemerides_1h.json", "time_tag", 30),
    # Alerts, watches and warnings: about the last 30 days.
    Feed("swpc_alerts", f"{SWPC}/products/alerts.json", "issue_datetime", 35),
    # NOAA Kp estimates and forecast: values change in place as estimates firm up.
    Feed("noaa_kp", f"{SWPC}/products/noaa-planetary-k-index.json", "time_tag", 10),
    Feed(
        "noaa_kp_forecast",
        f"{SWPC}/products/noaa-planetary-k-index-forecast.json",
        "time_tag",
        12,
    ),
    Feed("noaa_kp_1m", f"{SWPC}/json/planetary_k_index_1m.json", "time_tag", 2),
    # GFZ nowcasts: revised in place, so every revision is kept.
    Feed("gfz_kp", GFZ_JSON, "datetime", 4, _gfz_params("Kp")),
    Feed("gfz_hp30", GFZ_JSON, "datetime", 4, _gfz_params("Hp30")),
)
