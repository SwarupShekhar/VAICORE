import asyncio
import sys
from celery import Task
from celery.signals import worker_process_init

from celery_app import celery
from upload_log_db import update_log_status
from logger import get_logger

log = get_logger("vaidikai.tasks")

@worker_process_init.connect
def setup_worker_process(**kwargs):
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")


def _dispose_and_run(coro):
    """
    Dispose the shared async DB engine before running the coroutine.
    
    Each asyncio.run() creates a fresh event loop, but the module-level
    engine from database.py holds asyncpg connections bound to the previous
    loop. Disposing forces all stale connections to close so new ones are
    created on the current loop.
    """
    from database import engine
    engine.sync_engine.dispose(close=False)
    return asyncio.run(coro)


class PipelineTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        client_code = kwargs.get("client_code")
        original_filename = kwargs.get("original_filename")
        
        # Fallback to positional args if not in kwargs
        if not client_code and len(args) >= 2:
            client_code = args[1]
        if not original_filename and len(args) >= 4:
            original_filename = args[3]
            
        timestamp = kwargs.get("timestamp")
        if not timestamp and len(args) >= 5:
            timestamp = args[4]
            
        if client_code and original_filename:
            try:
                _dispose_and_run(update_log_status(
                    client_code=client_code,
                    filename=original_filename,
                    timestamp=timestamp,
                    status="dead_letter",
                    error=str(exc)
                ))
            except Exception as e:
                log.error(f"Failed to set dead_letter status for {original_filename}: {e}")

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_full_pipeline(self, *args, **kwargs):
    try:
        from main import run_full_pipeline
        _dispose_and_run(run_full_pipeline(*args, **kwargs))
    except Exception as exc:
        import traceback
        log.error(f"Full pipeline error: {traceback.format_exc()}")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_vad_pipeline(self, *args, **kwargs):
    try:
        from main import run_vad_pipeline
        _dispose_and_run(run_vad_pipeline(*args, **kwargs))
    except Exception as exc:
        import traceback
        log.error(f"VAD pipeline error: {traceback.format_exc()}")
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_jewelry_pipeline(self, *args, **kwargs):
    try:
        from main import run_jewelry_pipeline
        _dispose_and_run(run_jewelry_pipeline(*args, **kwargs))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_form_pipeline(self, *args, **kwargs):
    try:
        from main import run_form_pipeline
        _dispose_and_run(run_form_pipeline(*args, **kwargs))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_clickstream_pipeline(self, *args, **kwargs):
    try:
        from main import run_clickstream_pipeline
        _dispose_and_run(run_clickstream_pipeline(*args, **kwargs))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_transcript_pipeline(self, *args, **kwargs):
    try:
        from main import run_transcript_pipeline
        _dispose_and_run(run_transcript_pipeline(*args, **kwargs))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

@celery.task(bind=True, base=PipelineTask, max_retries=3)
def task_run_zip_batch_pipeline(self, *args, **kwargs):
    try:
        from main import run_zip_batch_pipeline
        _dispose_and_run(run_zip_batch_pipeline(*args, **kwargs))
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)
