import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool

from alembic import context

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Base so Alembic can discover the ORM models for autogenerate
from src.cost.tracker import Base  # noqa: E402

config = context.config

# Override sqlalchemy.url with DATABASE_URL env var when present
database_url = os.getenv("DATABASE_URL", "")
if database_url:
    # ── Supabase Transaction Pooler → Direct Connection conversion ────────────
    # Alembic requires a direct/session-mode connection for DDL migrations.
    # Transaction pooler (port 6543, pooler host) does NOT support DDL.
    # We auto-convert the pooler URL to the direct Supabase host URL.
    #   FROM: postgresql://postgres.XXXX:pwd@aws-0-*.pooler.supabase.com:6543/postgres
    #   TO:   postgresql://postgres:pwd@db.XXXX.supabase.co:5432/postgres
    if ".pooler.supabase.com" in database_url and ":6543/" in database_url:
        import re

        # Extract project ref from the username (postgres.PROJECT_REF)
        match = re.search(r"postgres\.([a-z]+):", database_url)
        if match:
            project_ref = match.group(1)
            # Rebuild URL pointing at the direct DB host
            database_url = re.sub(
                r"postgres\.[a-z]+:([^@]+)@[^/]+:\d+/",
                rf"postgres:\1@db.{project_ref}.supabase.co:5432/",
                database_url,
            )

    # Escape '%' to bypass configparser's interpolation syntax
    database_url = database_url.replace("%", "%%")
    config.set_main_option("sqlalchemy.url", database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations without a live DB connection (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
