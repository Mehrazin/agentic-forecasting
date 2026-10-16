"""Live FRED client via the vault surrogate (the API key never enters this process).

FRED's API reads the key from the ``api_key`` query parameter. The
``custom.fred`` connector holds it; the surrogate injects it at request
time. See ``scripts/fetch_fred.py`` for the curated bulk downloader.

Rate limits: 120 requests/minute per key — generous, but a 0.6s
client-side throttle keeps us polite.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from types import ModuleType

import pandas as pd

logger = logging.getLogger(__name__)

_FRED_HOST = "api.stlouisfed.org"
_FRED_BASE = f"https://{_FRED_HOST}/fred"
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_THROTTLE_S = 0.6

#: Location of the vault-surrogate helper (Muse VM connector layout).
#: Hoisted here (once, de-duplicated) so request paths never accumulate
#: duplicate sys.path entries.
_CONNECTOR_BIN = "/opt/muse/skills/custom-connector-setup/bin"
if _CONNECTOR_BIN not in sys.path:
    sys.path.insert(0, _CONNECTOR_BIN)


class FredUnavailableError(RuntimeError):
    """Raised when live FRED cannot be reached (no credential, no network)."""


def _dynamic_credentials() -> ModuleType:
    """Import the vault-surrogate helper module.

    The import stays lazy (via :func:`importlib.import_module`) so that
    importing ``td_one.fred`` never fails on machines without the Muse
    connector layout; callers get :class:`FredUnavailableError` instead.
    Loaded once via ``sys.modules`` caching.
    """
    try:
        return importlib.import_module("dynamic_credentials")
    except ImportError as exc:
        raise FredUnavailableError(
            "vault surrogate helper unavailable; is this running on the Muse VM?"
        ) from exc


def _surrogate_url(base: str, credential: str) -> str:
    """Attach the vault surrogate api_key query param to a FRED URL."""
    dc = _dynamic_credentials()
    url: str = dc.url_with_surrogate_query_param(
        base, credential, entry_name="access_token", allowed_hosts=[_FRED_HOST]
    )
    return url


class FredClient:
    """Minimal FRED API client (observations + series metadata)."""

    def __init__(self, credential: str = "custom.fred") -> None:
        """Initialize.

        Args:
            credential: Vault connector name holding the FRED API key.

        """
        self.credential = credential
        self._last_call_ts = 0.0

    def _get(self, path: str, params: dict[str, str]) -> dict[str, object]:
        dc = _dynamic_credentials()

        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < _THROTTLE_S:
            time.sleep(_THROTTLE_S - elapsed)
        query = urllib.parse.urlencode({**params, "file_type": "json"})
        try:
            url = _surrogate_url(f"{_FRED_BASE}/{path}?{query}", self.credential)
        except FredUnavailableError:
            raise
        except Exception as exc:
            raise FredUnavailableError(
                f"FRED credential {self.credential!r} unavailable in the vault: {exc}"
            ) from exc
        req = urllib.request.Request(url)
        req.add_header("User-Agent", _BROWSER_UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data: dict[str, object] = dc.read_json_response(resp)
        except FredUnavailableError:
            raise
        except Exception as exc:
            raise FredUnavailableError(f"FRED request failed: {exc}") from exc
        finally:
            self._last_call_ts = time.monotonic()
        return data

    def series_info(self, series_id: str) -> dict[str, object]:
        """Return FRED metadata for one series id (title, frequency, units...)."""
        data = self._get("series", {"series_id": series_id})
        seriess = data.get("seriess", [])
        if not isinstance(seriess, list) or not seriess:
            raise FredUnavailableError(f"FRED has no series {series_id!r}")
        first = seriess[0]
        if not isinstance(first, dict):
            raise FredUnavailableError(f"FRED returned malformed metadata for {series_id!r}")
        return dict(first)

    def observations(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
        vintage_dates: str | None = None,
    ) -> pd.DataFrame:
        """Fetch observations as a DataFrame with ``date`` and ``value`` columns.

        FRED's ``"."`` missing marker becomes NaN. ``start``/``end`` bound
        the observation window.

        Vintage warning: ``start``/``end`` filter by *observation* period
        and FRED returns the **latest vintage** by default — values for
        revised series (CPI, unemployment, Case-Shiller, ...) reflect later
        revisions, not what was known at ``end``. An observation-date cutoff
        alone is NOT vintage-correct. For a true as-of pull, pass
        ``vintage_dates`` (``"YYYY-MM-DD"`` or ``"start,end"``); see FRED's
        ``vintage_dates`` parameter.
        """
        if start is not None and end is not None and start > end:
            raise ValueError(f"start {start.isoformat()} is after end {end.isoformat()}")
        params: dict[str, str] = {"series_id": series_id}
        if start is not None:
            params["observation_start"] = start.isoformat()
        if end is not None:
            params["observation_end"] = end.isoformat()
        if vintage_dates is not None:
            params["vintage_dates"] = vintage_dates
        data = self._get("series/observations", params)
        if "error_code" in data:
            # FRED answers bad requests with {"error_code": ..., ...} and no
            # "observations" key — surface it instead of returning silence.
            raise FredUnavailableError(
                f"FRED observations request failed for {series_id!r}: "
                f"{data.get('error_message', data.get('error_code'))}"
            )
        rows_raw = data.get("observations", [])
        if not isinstance(rows_raw, list):
            raise FredUnavailableError(f"FRED returned malformed observations for {series_id!r}")
        rows = [
            (r["date"], float(r["value"]) if r["value"] != "." else float("nan")) for r in rows_raw
        ]
        df = pd.DataFrame(rows, columns=["date", "value"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df


def get_series_live(
    series_id: str,
    start: date,
    end: date,
    api_key: str | None = None,  # noqa: ARG001 - legacy signature; vault is used
) -> pd.Series:
    """Fetch a live FRED series as a date-indexed Series of floats.

    Args:
        series_id: FRED series id (e.g. ``"DGS5"``).
        start: First observation date (inclusive).
        end: Last observation date (inclusive) — the PIT cutoff.
        api_key: Ignored; the key comes from the ``custom.fred`` vault
            connector. Kept for backwards compatibility.

    Raises:
        FredUnavailableError: No vault credential or no network.

    """
    df = FredClient().observations(series_id, start=start, end=end)
    if df.empty:
        return pd.Series(dtype=float)
    return pd.Series(df["value"].to_numpy(), index=pd.to_datetime(df["date"]), name=series_id)
