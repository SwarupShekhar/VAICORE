"""add revoked_tokens table (JWT revocation)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-23

"""
from alembic import op
import sqlalchemy as sa

revision = "c3d4e5f6a7b8"
down_revision = "b2c3d4e5f6a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent: the table may already exist from a manual hotfix on the
    # server (it was missing because no migration originally created it).
    bind = op.get_bind()
    if sa.inspect(bind).has_table("revoked_tokens"):
        return
    op.create_table(
        "revoked_tokens",
        sa.Column("jti", sa.String(), primary_key=True, nullable=False),
        sa.Column(
            "revoked_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("revoked_tokens")
