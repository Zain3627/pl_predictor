"""
Migrate MLflow data from local SQLite (mlflow.db) to the PostgreSQL database
configured in MLFLOW_TRACKING_URI (src/.env).

Usage:
    python migrate_mlflow_to_postgres.py [--dry-run]
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
SQLITE_PATH = PROJECT_ROOT / "mlflow.db"
ENV_PATH = PROJECT_ROOT / "src" / ".env"

# Tables migrated in dependency order so FK constraints are never violated.
# Tables that exist in MLflow's schema but carry no data we care about
# (e.g. jobs, endpoints, scorers, …) are skipped — they will remain empty
# in Postgres, which is fine.
ORDERED_TABLES = [
    "alembic_version",
    "experiments",
    "experiment_tags",
    "runs",
    "params",
    "metrics",
    "latest_metrics",
    "tags",
    "datasets",
    "inputs",
    "input_tags",
    "registered_models",
    "registered_model_tags",
    "model_versions",
    "model_version_tags",
    "registered_model_aliases",
    "trace_info",
    "trace_request_metadata",
    "trace_tags",
    "span_metrics",
    "spans",
    "trace_metrics",
    "logged_models",
    "logged_model_params",
    "logged_model_metrics",
    "logged_model_tags",
]


def load_postgres_uri() -> str:
    load_dotenv(ENV_PATH)
    uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("postgresql"):
        sys.exit(
            f"ERROR: MLFLOW_TRACKING_URI is '{uri}'. "
            "Expected a postgresql:// URI. Check src/.env."
        )
    return uri


def get_sqlite_tables(sqlite_engine) -> set[str]:
    return set(inspect(sqlite_engine).get_table_names())


def get_postgres_tables(pg_engine) -> set[str]:
    return set(inspect(pg_engine).get_table_names())


def get_bool_columns(table: str, pg_engine) -> set[str]:
    """Return column names that are boolean type in the PostgreSQL schema."""
    cols = inspect(pg_engine).get_columns(table)
    from sqlalchemy import Boolean
    return {c["name"] for c in cols if isinstance(c["type"], Boolean)}


def coerce_row(row: dict, bool_cols: set[str]) -> dict:
    """Convert SQLite integer 0/1 values to Python bools for boolean columns."""
    result = dict(row)
    for col in bool_cols:
        if col in result and result[col] is not None:
            result[col] = bool(result[col])
    return result


def migrate_table(table: str, src_engine, dst_engine, dry_run: bool) -> int:
    """Copy all rows from *table* in src to dst. Returns number of rows inserted."""
    with src_engine.connect() as src_conn:
        rows = src_conn.execute(text(f"SELECT * FROM {table}")).mappings().all()

    if not rows:
        print(f"  {table}: (empty — skipped)")
        return 0

    if dry_run:
        print(f"  {table}: {len(rows)} row(s) [dry-run, not written]")
        return 0

    bool_cols = get_bool_columns(table, dst_engine)

    # Build an INSERT … ON CONFLICT DO NOTHING so re-runs are idempotent.
    cols = list(rows[0].keys())
    col_list = ", ".join(f'"{c}"' for c in cols)
    val_list = ", ".join(f":{c}" for c in cols)
    sql = text(
        f'INSERT INTO "{table}" ({col_list}) VALUES ({val_list}) '
        "ON CONFLICT DO NOTHING"
    )

    inserted = 0
    with dst_engine.begin() as dst_conn:
        for row in rows:
            result = dst_conn.execute(sql, coerce_row(dict(row), bool_cols))
            inserted += result.rowcount

    print(f"  {table}: {len(rows)} row(s) read, {inserted} inserted")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Migrate mlflow.db → PostgreSQL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read SQLite and report what would be migrated, but write nothing.",
    )
    args = parser.parse_args()

    if not SQLITE_PATH.exists():
        sys.exit(f"ERROR: {SQLITE_PATH} not found.")

    pg_uri = load_postgres_uri()

    print(f"Source : {SQLITE_PATH}")
    print(f"Dest   : {pg_uri[:60]}…")
    if args.dry_run:
        print("Mode   : DRY RUN — no data will be written\n")
    else:
        print("Mode   : LIVE — data will be written\n")

    sqlite_engine = create_engine(f"sqlite:///{SQLITE_PATH}")
    pg_engine = create_engine(pg_uri)

    sqlite_tables = get_sqlite_tables(sqlite_engine)
    pg_tables = get_postgres_tables(pg_engine)

    total_inserted = 0
    for table in ORDERED_TABLES:
        if table not in sqlite_tables:
            print(f"  {table}: not in SQLite — skipped")
            continue
        if table not in pg_tables:
            print(f"  {table}: not in PostgreSQL — skipped (schema mismatch?)")
            continue
        total_inserted += migrate_table(table, sqlite_engine, pg_engine, args.dry_run)

    print(f"\nDone. {total_inserted} row(s) inserted into PostgreSQL.")


if __name__ == "__main__":
    main()
