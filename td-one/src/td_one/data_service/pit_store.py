"""Point-in-time DuckDB store for mortgage data.

Enforces strict temporal boundaries:
- mortgage_history: snapshot_date <= run_date
- renewal_outcomes: record_asof <= run_date (and outcome_observed_at <= run_date
  for resolved outcomes; pending otherwise)

This is the TemporalGuard: no post-run_date data ever leaks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from td_one.schemas import MortgageHistoryRecord, RenewalOutcomeRecord


class PITStore:
    """DuckDB-backed point-in-time query engine."""

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the store.

        Args:
            db_path: Path to DuckDB file (created if not exists).

        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if not exists."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mortgage_history (
                customer_id VARCHAR,
                mortgage_id VARCHAR,
                snapshot_date DATE,
                province VARCHAR,
                mortgage_rate_pct DOUBLE,
                td_liquid_deposits_cad DOUBLE,
                credit_score_band VARCHAR,
                household_income_cad DOUBLE,
                children_under6 INTEGER,
                children_6to17 INTEGER,
                mortgage_balance_cad DOUBLE,
                insured_flag BOOLEAN,
                PRIMARY KEY (mortgage_id, snapshot_date)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS renewal_outcomes (
                customer_id VARCHAR,
                mortgage_id VARCHAR,
                renewal_episode_id VARCHAR,
                scheduled_maturity_date DATE,
                outcome VARCHAR,
                switched_lender_flag BOOLEAN,
                event_date DATE,
                outcome_observed_at DATE,
                record_asof DATE,
                PRIMARY KEY (mortgage_id, renewal_episode_id, record_asof)
            )
            """
        )

    def ingest_history(self, records: list[MortgageHistoryRecord]) -> None:
        """Bulk ingest history records.

        Args:
            records: Validated mortgage history records.

        """
        if not records:
            return
        rows = [
            (
                r.customer_id,
                r.mortgage_id,
                r.snapshot_date.isoformat(),
                r.province,
                r.mortgage_rate_pct,
                r.td_liquid_deposits_cad,
                r.credit_score_band,
                r.household_income_cad,
                r.children_under6,
                r.children_6to17,
                r.mortgage_balance_cad,
                r.insured_flag,
            )
            for r in records
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO mortgage_history VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )

    def ingest_outcomes(self, records: list[RenewalOutcomeRecord]) -> None:
        """Bulk ingest renewal outcome records.

        Args:
            records: Validated renewal outcome records.

        """
        if not records:
            return
        rows = [
            (
                r.customer_id,
                r.mortgage_id,
                r.renewal_episode_id,
                r.scheduled_maturity_date.isoformat(),
                r.outcome,
                r.switched_lender_flag,
                r.event_date.isoformat() if r.event_date else None,
                r.outcome_observed_at.isoformat() if r.outcome_observed_at else None,
                r.record_asof.isoformat(),
            )
            for r in records
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO renewal_outcomes VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            rows,
        )

    def get_outcomes(self, run_date: date) -> pd.DataFrame:
        """Get outcomes visible at point-in-time.

        Enforces both guards:
        - record_asof <= run_date (row existed)
        - outcome_observed_at <= run_date (outcome was known; otherwise the
          row is masked even if record_asof passed)

        Args:
            run_date: Point-in-time cutoff (inclusive).

        Returns:
            DataFrame of visible outcome records.

        """
        return self._conn.execute(
            """
            SELECT * FROM renewal_outcomes
            WHERE record_asof <= ?
              AND (outcome_observed_at IS NULL OR outcome_observed_at <= ?)
            """,
            [run_date.isoformat(), run_date.isoformat()],
        ).df()

    def get_history(self, run_date: date) -> pd.DataFrame:
        """Get all history with snapshot_date <= run_date.

        Args:
            run_date: Point-in-time cutoff (inclusive).

        Returns:
            DataFrame of history records.

        """
        return self._conn.execute(
            """
            SELECT * FROM mortgage_history
            WHERE snapshot_date <= ?
            """,
            [run_date.isoformat()],
        ).df()

    def get_segment(
        self,
        run_date: date,
        province: str | None = None,
        max_rate_pct: float | None = None,
        min_children: int | None = None,
    ) -> pd.DataFrame:
        """Get segment-filtered history at point-in-time.

        Args:
            run_date: PIT cutoff.
            province: Filter by province (e.g. "ON").
            max_rate_pct: Filter rate <= value (for low-rate cohort).
            min_children: Filter total children >= value (for family cohort).

        Returns:
            Filtered DataFrame.

        """
        query = "SELECT * FROM mortgage_history WHERE snapshot_date <= ?"
        params: list[str | float | int] = [run_date.isoformat()]

        if province:
            query += " AND province = ?"
            params.append(province)
        if max_rate_pct is not None:
            query += " AND mortgage_rate_pct <= ?"
            params.append(max_rate_pct)
        if min_children is not None:
            query += " AND (children_under6 + children_6to17) >= ?"
            params.append(min_children)

        return self._conn.execute(query, params).df()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
