import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Double,
    Enum,
    ForeignKey,
    Integer,
    String,
    event,
    DDL,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ===========================================================================
# 1. Custom SQLAlchemy Enum Types
# ===========================================================================

class JobStatus(enum.Enum):
    UPLOADED = "Uploaded"
    PROCESSING = "Processing"
    TRANSCRIBING = "Transcribing"
    IN_REVIEW = "In Review"
    DELIVERED = "Delivered"
    ERROR = "Error"
    FAILED = "Failed"


class JobCategory(enum.Enum):
    AUDIO = "audio"
    JEWELRY = "jewelry"
    HOUSING = "housing"
    BUSINESS = "business"
    FORM = "form"
    CLICKSTREAM = "clickstream"
    TRANSCRIPT = "transcript"
    AUTO = "auto"


# ===========================================================================
# 2. Declarative Base
# ===========================================================================

class Base(DeclarativeBase):
    """SQLAlchemy 2.x Declarative Base with JSONB and Postgres UUID mapping."""
    type_annotation_map = {
        dict: JSONB,
    }


# ===========================================================================
# 3. Database Models
# ===========================================================================

class Client(Base):
    """
    Stores tenant/client profiles, dashboard session keys, custom UI timeline labels,
    and client contact metadata.
    """
    __tablename__ = "clients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_code: Mapped[str] = mapped_column(
        String(50), unique=True, nullable=False, index=True
    )
    client_name: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    contact_email: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    active: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )
    access_token: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    upload_token: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True, index=True
    )
    project_ids: Mapped[Optional[dict]] = mapped_column(
        JSONB, default=dict, nullable=True
    )
    role_labels: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String(100)), default=["Agent", "Customer"], nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    download_tokens: Mapped[List["ClientDownloadToken"]] = relationship(
        "ClientDownloadToken", back_populates="client", cascade="all, delete-orphan"
    )
    upload_logs: Mapped[List["UploadLog"]] = relationship(
        "UploadLog", back_populates="client", cascade="all, delete-orphan"
    )
    collateral_signatures: Mapped[List["CollateralSignature"]] = relationship(
        "CollateralSignature", back_populates="client", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Client(id={self.id}, client_code='{self.client_code}', "
            f"client_name='{self.client_name}', active={self.active})>"
        )


class ClientDownloadToken(Base):
    """
    Manages secure, expiring, or permanent file delivery links for packaged annotated bundles.
    """
    __tablename__ = "client_download_tokens"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    download_token: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False, index=True
    )
    blob_path: Mapped[str] = mapped_column(
        String, nullable=False
    )
    label: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    client: Mapped["Client"] = relationship(
        "Client", back_populates="download_tokens"
    )

    def __repr__(self) -> str:
        return (
            f"<ClientDownloadToken(id={self.id}, client_id={self.client_id}, "
            f"label='{self.label}', expires_at={self.expires_at})>"
        )


class UploadLog(Base):
    """
    Tracks customer raw intakes, file processing stages, model pipeline analytics,
    and execution errors.
    """
    __tablename__ = "upload_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    file_size: Mapped[int] = mapped_column(
        BigInteger, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status_enum", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        default=JobStatus.UPLOADED,
        nullable=False,
    )
    category: Mapped[JobCategory] = mapped_column(
        Enum(JobCategory, name="job_category_enum", native_enum=True, values_callable=lambda x: [e.value for e in x]),
        default=JobCategory.AUTO,
        nullable=False,
    )
    language: Mapped[Optional[str]] = mapped_column(
        String(10), nullable=True
    )
    is_batch: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    batch_id: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True, index=True
    )
    parent_zip: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    sub_blob_name: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    error: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    labelstudio_error: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    predictions_count: Mapped[Optional[int]] = mapped_column(
        Integer, default=0, nullable=True
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    client: Mapped["Client"] = relationship(
        "Client", back_populates="upload_logs"
    )

    def __repr__(self) -> str:
        return (
            f"<UploadLog(id={self.id}, filename='{self.filename}', "
            f"status='{self.status.value}', updated_at={self.updated_at})>"
        )


class CollateralSignature(Base):
    """
    Manages structural 3D vectors and visual fingerprints of jewelry items
    or facades to fuel visual deduplication and fraud prevention checkups.
    """
    __tablename__ = "collateral_signatures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    task_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True
    )
    item_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    image_file: Mapped[str] = mapped_column(
        String(255), nullable=False
    )
    category: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    phash: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )
    hu_moments: Mapped[List[float]] = mapped_column(
        ARRAY(Double), nullable=False
    )
    color_histogram: Mapped[List[float]] = mapped_column(
        ARRAY(Double), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    client: Mapped["Client"] = relationship(
        "Client", back_populates="collateral_signatures"
    )

    # Bounded Float Array CHECK Constraints
    __table_args__ = (
        CheckConstraint(
            "array_length(hu_moments, 1) = 7", name="chk_hu_moments_length"
        ),
        CheckConstraint(
            "array_length(color_histogram, 1) = 144",
            name="chk_color_histogram_length",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<CollateralSignature(id={self.id}, task_id={self.task_id}, "
            f"category='{self.category}', phash='{self.phash}')>"
        )


# ===========================================================================
# 4. DDL Event Listeners for Alembic and Metadata Triggers
# ===========================================================================

# DDL definitions to handle state transition updates automatically on Postgres
create_trigger_func_ddl = DDL("""
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';
""")

create_trigger_ddl = DDL("""
CREATE TRIGGER trigger_update_upload_logs_timestamp
BEFORE UPDATE ON upload_logs
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
""")

# Bind event listeners specifically to the UploadLog table during creation
event.listen(
    UploadLog.__table__,
    "after_create",
    create_trigger_func_ddl.execute_if(dialect="postgresql"),
)
event.listen(
    UploadLog.__table__,
    "after_create",
    create_trigger_ddl.execute_if(dialect="postgresql"),
)
