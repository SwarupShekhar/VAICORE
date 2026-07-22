"""add user_sessions and audit_logs tables (security hardening)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-07-22

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b2c3d4e5f6a7"
down_revision = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("jti", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "revoked", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
            name="fk_user_sessions_user_id",
        ),
    )
    op.create_index(
        "ix_user_sessions_jti", "user_sessions", ["jti"], unique=True
    )
    op.create_index(
        "ix_user_sessions_user_id", "user_sessions", ["user_id"]
    )

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.String(length=100), nullable=False),
        sa.Column("target", sa.String(length=255), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_audit_logs_actor_user_id", "audit_logs", ["actor_user_id"]
    )
    op.create_index(
        "ix_audit_logs_created_at", "audit_logs", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_index("ix_user_sessions_jti", table_name="user_sessions")
    op.drop_table("user_sessions")
