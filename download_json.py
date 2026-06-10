import os
import sys
import json
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

container_client = blob_service_client.get_container_client("client-delivery")

blob_name = "CLIENT002/CLIENT002_20260609_131001_7608087163421399032_1_7608087172011333688_transcript.json"

blob_client = blob_service_client.get_blob_client(container="client-delivery", blob=blob_name)
data = blob_client.download_blob().readall()
parsed = json.loads(data)

print(json.dumps(parsed, indent=2, ensure_ascii=False))
