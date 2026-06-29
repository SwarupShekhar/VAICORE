import asyncio
import os
from datetime import datetime, timedelta
from azure.storage.blob.aio import BlobServiceClient
from logger import get_logger
from upload_log_db import load_upload_log, update_log_status

log = get_logger("vaidikai.auto_purge")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

async def run_purge():
    if not AZURE_STORAGE_CONNECTION_STRING:
        log.warning("AZURE_STORAGE_CONNECTION_STRING is missing. Skipping auto-purge.")
        return

    log.info("Starting auto-purge cycle...")
    
    # 4 days ago
    cutoff_date = datetime.now() - timedelta(days=4)
    
    logs = await load_upload_log()
    
    # Filter logs that are 'Delivered' and older than cutoff_date
    logs_to_purge = []
    for entry in logs:
        if entry.get("status") == "Delivered" and entry.get("timestamp"):
            try:
                uploaded_at = datetime.strptime(entry["timestamp"], "%Y%m%d_%H%M%S")
                if uploaded_at < cutoff_date:
                    logs_to_purge.append(entry)
            except Exception as e:
                pass

    if not logs_to_purge:
        log.info("No files eligible for purging.")
        return
        
    log.info(f"Found {len(logs_to_purge)} entries to purge.")
    
    async with BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING) as blob_service_client:
        container_raw = blob_service_client.get_container_client("raw")
        container_processing = blob_service_client.get_container_client("processing")
        container_delivery = blob_service_client.get_container_client("client-delivery")
        
        for entry in logs_to_purge:
            client_code = entry.get("client_code")
            filename = entry.get("filename")
            
            log.info(f"Purging files for {client_code}/{filename}...")
            
            # Possible blobs to delete
            blobs_to_delete = [
                # Raw container
                (container_raw, f"{client_code}/{filename}"),
                # Processing container
                (container_processing, f"{client_code}/{filename}"),
                (container_processing, f"{client_code}/{filename}_transcript.json"),
                # Deliver container
                (container_delivery, f"{client_code}/{filename}"),
                (container_delivery, f"{client_code}/{filename.rsplit('.', 1)[0]}.xlsx"),
            ]
            
            # Delivery zip if batch
            batch_id = entry.get("batch_id")
            if batch_id:
                blobs_to_delete.append((container_delivery, f"{client_code}/{batch_id}_delivered.zip"))
            
            for container_client, blob_name in blobs_to_delete:
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    if await blob_client.exists():
                        await blob_client.delete_blob()
                        log.info(f"Deleted blob: {blob_name}")
                except Exception as e:
                    log.warning(f"Failed to delete blob {blob_name}: {e}")
            
            # Finally, update the database ledger to 'Purged'
            await update_log_status(client_code, filename, entry.get("timestamp"), "Purged")
            log.info(f"Ledger updated to 'Purged' for {client_code}/{filename}")

        # Clean up any manual delivery packages (delivery_package_*.zip) older than 4 days
        log.info("Scanning for old manual delivery zip packages...")
        try:
            async for blob in container_delivery.list_blobs():
                if "delivery_package_" in blob.name and blob.name.endswith(".zip"):
                    if blob.last_modified.replace(tzinfo=None) < cutoff_date:
                        log.info(f"Deleting old manual delivery zip: {blob.name}")
                        try:
                            blob_client = container_delivery.get_blob_client(blob.name)
                            await blob_client.delete_blob()
                        except Exception as e:
                            log.warning(f"Failed to delete old zip {blob.name}: {e}")
        except Exception as e:
            log.warning(f"Failed to scan and delete manual delivery zips: {e}")

