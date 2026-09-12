"""Create documents metadata table.

Revision ID: 9c4e78f3d2ab
Revises: 54eed185bd51
Create Date: 2026-09-10 13:05:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9c4e78f3d2ab"
down_revision: Union[str, Sequence[str], None] = "54eed185bd51"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("content_type", sa.String(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("storage_bucket", sa.String(), nullable=False),
        sa.Column("storage_path", sa.String(), nullable=False, unique=True),
    )
    op.create_index("ix_documents_user_active", "documents", ["user_id", "deleted_at"])
    op.create_index("ix_documents_filename", "documents", ["filename"])


def downgrade() -> None:
    op.drop_index("ix_documents_filename", table_name="documents")
    op.drop_index("ix_documents_user_active", table_name="documents")
    op.drop_table("documents")
