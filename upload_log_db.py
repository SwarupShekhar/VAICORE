import uuid
from typing import Optional
from datetime import datetime
from sqlalchemy import select, update, delete
from database import get_db_session
from models import Client, UploadLog, JobStatus, JobCategory

LEGACY_STATUS_MAP = {
    "Completed": "Delivered",
    "Failed (Audio)": "Failed",
    "Failed (Label Studio)": "Failed", 
    "Interrupted (System Crash)": "Failed",
    None: "Uploaded",
}

async def get_client_id_for_code(client_code: str) -> Optional[uuid.UUID]:
    async with get_db_session() as session:
        stmt = select(Client.id).where(Client.client_code == client_code)
        res = await session.execute(stmt)
        return res.scalar_one_or_none()

async def load_upload_log(client_code: Optional[str] = None) -> list:
    async with get_db_session() as session:
        stmt = select(UploadLog, Client.client_code).join(Client, UploadLog.client_id == Client.id)
        if client_code:
            stmt = stmt.where(Client.client_code == client_code)
        stmt = stmt.order_by(UploadLog.uploaded_at.asc())
        
        result = await session.execute(stmt)
        records = []
        for row in result:
            log = row[0]
            code = row[1]
            records.append({
                "client_code": code,
                "filename": log.filename,
                "file_size": log.file_size,
                "status": log.status.value if log.status else None,
                "category": log.category.value if log.category else None,
                "language": log.language,
                "is_batch": log.is_batch,
                "batch_id": log.batch_id,
                "parent_zip": log.parent_zip,
                "sub_blob_name": log.sub_blob_name,
                "error": log.error,
                "labelstudio_error": log.labelstudio_error,
                "predictions_count": log.predictions_count,
                "timestamp": log.uploaded_at.strftime("%Y%m%d_%H%M%S") if log.uploaded_at else None,
            })
        return records

async def append_upload_log(entry: dict) -> None:
    async with get_db_session() as session:
        client_code = entry.get("client_code")
        client_stmt = select(Client.id).where(Client.client_code == client_code)
        client_res = await session.execute(client_stmt)
        client_id = client_res.scalar_one_or_none()
        if not client_id:
            raise ValueError(f"Client code {client_code} not found")

        entry_status = entry.get("status")
        entry_status = LEGACY_STATUS_MAP.get(entry_status, entry_status)
        status_enum = JobStatus(entry_status) if entry_status else JobStatus.UPLOADED

        category_str = entry.get("category")
        category_enum = JobCategory(category_str) if category_str else JobCategory.AUTO

        uploaded_at_val = datetime.now()
        timestamp_str = entry.get("timestamp")
        if timestamp_str:
            try:
                uploaded_at_val = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except Exception:
                pass

        new_log = UploadLog(
            client_id=client_id,
            filename=entry.get("filename"),
            file_size=entry.get("file_size", 0),
            status=status_enum,
            category=category_enum,
            language=entry.get("language"),
            is_batch=entry.get("is_batch", False),
            batch_id=entry.get("batch_id"),
            parent_zip=entry.get("parent_zip"),
            sub_blob_name=entry.get("sub_blob_name"),
            error=entry.get("error"),
            labelstudio_error=entry.get("labelstudio_error"),
            predictions_count=entry.get("predictions_count", 0),
            uploaded_at=uploaded_at_val,
        )
        session.add(new_log)
        await session.commit()

async def update_log_status(client_code: str, filename: str, timestamp: str, status: str, **kwargs) -> None:
    async with get_db_session() as session:
        client_stmt = select(Client.id).where(Client.client_code == client_code)
        client_res = await session.execute(client_stmt)
        client_id = client_res.scalar_one_or_none()
        if not client_id:
            return

        uploaded_at_val = None
        if timestamp:
            try:
                uploaded_at_val = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            except Exception:
                pass

        stmt = (
            select(UploadLog)
            .where(UploadLog.client_id == client_id)
            .where(UploadLog.filename == filename)
        )
        if uploaded_at_val:
            stmt = stmt.where(UploadLog.uploaded_at == uploaded_at_val)
        else:
            stmt = stmt.order_by(UploadLog.uploaded_at.desc())

        res = await session.execute(stmt)
        log_entry = res.scalars().first()
        if not log_entry:
            return

        status = LEGACY_STATUS_MAP.get(status, status)
        log_entry.status = JobStatus(status) if status else JobStatus.UPLOADED

        for key, value in kwargs.items():
            if key == "category":
                log_entry.category = JobCategory(value) if value else JobCategory.AUTO
            elif hasattr(log_entry, key):
                setattr(log_entry, key, value)

        await session.commit()

async def delete_log_entry(client_code: str, filename: str) -> None:
    async with get_db_session() as session:
        client_id = await get_client_id_for_code(client_code)
        if not client_id:
            return
        
        stmt = (
            delete(UploadLog)
            .where(UploadLog.client_id == client_id)
            .where(UploadLog.filename == filename)
        )
        await session.execute(stmt)
        await session.commit()
