"""initial schema — create query_logs table

Revision ID: 68f612722729
Revises:
Create Date: 2026-07-14 07:10:11.747787
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "68f612722729"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(), nullable=True),
        sa.Column("query", sa.String(), nullable=False),
        sa.Column("query_hash", sa.String(), nullable=True),
        sa.Column("query_length", sa.Integer(), nullable=True),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("complexity", sa.String(), nullable=True),
        sa.Column("strategy", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost", sa.Float(), nullable=True),
        sa.Column("latency", sa.Float(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
    )
    # Index for time-range cost queries (the most common query pattern)
    op.create_index("ix_query_logs_timestamp", "query_logs", ["timestamp"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_timestamp", table_name="query_logs")
    op.drop_table("query_logs")
