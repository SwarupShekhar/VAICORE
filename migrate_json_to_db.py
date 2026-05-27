import json
import asyncio
from datetime import datetime
from sqlalchemy import select
from database import get_db_session
from models import Client, ClientDownloadToken, UploadLog, JobStatus, JobCategory
from upload_log_db import LEGACY_STATUS_MAP

async def migrate_data():
    print("Starting data migration...")
    
    # 1. Migrate Clients and Download Tokens
    try:
        with open("clients.json", "r") as f:
            clients_data = json.load(f)
    except FileNotFoundError:
        print("clients.json not found, skipping clients migration.")
        clients_data = {}

    clients_migrated = 0
    tokens_migrated = 0

    async with get_db_session() as session:
        for access_token, info in clients_data.items():
            client_code = info.get("client_code")
            if not client_code:
                continue

            # Check if client already exists
            stmt = select(Client).where(Client.client_code == client_code)
            res = await session.execute(stmt)
            db_client = res.scalar_one_or_none()

            created_at_dt = datetime.now()
            created_at_str = info.get("created_at")
            if created_at_str:
                try:
                    created_at_dt = datetime.strptime(created_at_str, "%Y-%m-%d")
                except Exception:
                    pass

            if not db_client:
                db_client = Client(
                    client_code=client_code,
                    client_name=info.get("client_name", "Unknown"),
                    contact_email=info.get("contact_email"),
                    active=info.get("active", True),
                    access_token=access_token,
                    upload_token=info.get("upload_token"),
                    project_ids=info.get("project_ids", {}),
                    created_at=created_at_dt
                )
                session.add(db_client)
                await session.flush()  # Populates db_client.id
                clients_migrated += 1
                print(f"Added client: {client_code}")
            else:
                print(f"Client {client_code} already exists, skipping client insert.")

            # Migrate download tokens
            download_tokens = info.get("download_tokens", {})
            for token, t_info in download_tokens.items():
                t_stmt = select(ClientDownloadToken).where(ClientDownloadToken.download_token == token)
                t_res = await session.execute(t_stmt)
                db_token = t_res.scalar_one_or_none()

                if not db_token:
                    t_created_dt = datetime.now()
                    t_created_str = t_info.get("created_at")
                    if t_created_str:
                        try:
                            # Format: 2026-05-13T07:52:39.388865
                            t_created_dt = datetime.strptime(t_created_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
                        except Exception:
                            pass

                    new_token = ClientDownloadToken(
                        client_id=db_client.id,
                        download_token=token,
                        blob_path=t_info.get("blob_path", ""),
                        label=t_info.get("label"),
                        created_at=t_created_dt
                    )
                    session.add(new_token)
                    tokens_migrated += 1
                    print(f"  Added download token: {token[:8]}...")
        await session.commit()

    # 2. Migrate Upload Logs
    try:
        with open("upload_log.json", "r") as f:
            logs_data = json.load(f)
    except FileNotFoundError:
        print("upload_log.json not found, skipping logs migration.")
        logs_data = []

    logs_migrated = 0

    async with get_db_session() as session:
        for entry in logs_data:
            client_code = entry.get("client_code")
            filename = entry.get("filename")
            if not client_code or not filename:
                continue

            # Resolve client_code to client_id
            client_stmt = select(Client).where(Client.client_code == client_code)
            client_res = await session.execute(client_stmt)
            db_client = client_res.scalar_one_or_none()
            if not db_client:
                print(f"Warning: Client {client_code} not found for log {filename}, skipping.")
                continue

            # Parse uploaded_at
            uploaded_at_val = datetime.now()
            timestamp_str = entry.get("timestamp")
            if timestamp_str:
                try:
                    uploaded_at_val = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except Exception:
                    pass

            # Check if this log entry already exists
            log_stmt = select(UploadLog).where(
                UploadLog.client_id == db_client.id,
                UploadLog.filename == filename,
                UploadLog.uploaded_at == uploaded_at_val
            )
            log_res = await session.execute(log_stmt)
            db_log = log_res.scalar_one_or_none()

            if not db_log:
                entry_status = entry.get("status")
                entry_status = LEGACY_STATUS_MAP.get(entry_status, entry_status)
                status_enum = JobStatus(entry_status) if entry_status else JobStatus.UPLOADED

                category_str = entry.get("category")
                category_enum = JobCategory(category_str) if category_str else JobCategory.AUTO

                new_log = UploadLog(
                    client_id=db_client.id,
                    filename=filename,
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
                    uploaded_at=uploaded_at_val
                )
                session.add(new_log)
                logs_migrated += 1
            
        await session.commit()

    print("\nMigration Completed Successfully!")
    print(f"Clients Migrated: {clients_migrated}")
    print(f"Download Tokens Migrated: {tokens_migrated}")
    print(f"Upload Logs Migrated: {logs_migrated}")

if __name__ == "__main__":
    asyncio.run(migrate_data())
