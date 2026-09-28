"""Scope query telemetry by authenticated user.

Revision ID: c1f4a2e6b9d0
Revises: 9c4e78f3d2ab
Create Date: 2026-09-28 08:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1f4a2e6b9d0"
down_revision: Union[str, Sequence[str], None] = "9c4e78f3d2ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("query_logs", sa.Column("user_id", sa.String(), nullable=True))
    op.create_index(
        "ix_query_logs_user_timestamp",
        "query_logs",
        ["user_id", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_query_logs_user_timestamp", table_name="query_logs")
    op.drop_column("query_logs", "user_id")
