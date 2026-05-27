import os
from celery import Celery

VALKEY_URL = os.getenv("VALKEY_URL", "redis://valkey:6379/0")

celery = Celery(
    "vaidikai",
    broker=VALKEY_URL,
    backend=VALKEY_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    imports=("tasks",),
)
