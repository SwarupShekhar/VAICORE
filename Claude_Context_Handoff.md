# Vaidik AI Portal - VAD Pipeline Debugging Context

## The Goal
Fix the `VAD` (Voice as Data) pipeline which processes uploaded audio files. The system runs on a Vultr server via Docker Compose. The pipeline utilizes Celery workers to asynchronously download files from Azure Blob, extract audio, transcribe via a GPU endpoint (initially Groq, now RunPod), segment the text, and upload the transcript and audio clips to Label Studio.

## Fixed Issues So Far

1. **Celery Worker / Database Connection Crash**
   - **Error:** `RuntimeError: got Future <Future pending> attached to a different loop`
   - **Cause:** `asyncpg` combined with Celery. Each task called `asyncio.run(...)` which created a new event loop. However, the database engine (`database.py`) was a global singleton holding a connection pool. When a new task started, it grabbed an existing database connection bound to the old, destroyed event loop.
   - **Fix Applied:** Modified `tasks.py` to use a `_dispose_and_run()` wrapper. Before calling `asyncio.run()`, we now explicitly call `engine.sync_engine.dispose(close=False)` to drop all stale connections. The worker now successfully begins processing files without crashing on the database queries.

2. **Missing Environment Variables in Worker**
   - **Error:** `RUNPOD_API_KEY or RUNPOD_WHISPER_ENDPOINT not set in .env`
   - **Cause:** The server's `.env` file was missing `RUNPOD_WHISPER_ENDPOINT`, and the `celery-worker` container had not been recreated to pick up `.env` changes.
   - **Fix Applied:** Added the variable to the server's `.env` and ran `docker compose up -d --force-recreate celery-worker`.

3. **RunPod Whisper 400 Bad Request**
   - **Error:** `HTTP Request: POST https://api.runpod.ai/v2/mguwbmywfpmz0f/openai/v1/audio/transcriptions "HTTP/1.1 400 Bad Request" - invalid request body`
   - **Cause:** The codebase was previously configured for Groq, which fully supports the OpenAI API spec. The code sent `timestamp_granularities=["segment"]`, `temperature=0`, and `prompt` parameters. The RunPod serverless vLLM/Whisper endpoint is only *partially* OpenAI compatible and rejects these fields as invalid.
   - **Fix Applied:** Modified `vad_processor.py` (mono path) and `processor.py` (stereo path) to strip `timestamp_granularities`, `temperature`, and `prompt` when the client's `base_url` points to RunPod. 

## Current Status (Where Claude Needs to Pick Up)

The pipeline is still failing after applying the RunPod parameter fix.
**Crucially, the exact error log of the newest failure has not been captured yet.**

### What Claude needs to do immediately:
1. Log into the Vultr server via SSH: `ssh root@139.84.163.249`
2. Navigate to the directory: `cd /opt/vaidikai-portal/vaidikai-portal`
3. Trigger a file upload via the portal UI.
4. Immediately check the celery worker logs to capture the new stack trace:
   `docker compose logs celery-worker -f --tail 50`
5. Based on the logs, fix the next bottleneck.

### Potential Suspects for the current failure:
1. **Docker stale cache:** The celery-worker might still be running the `.pyc` compiled bytecode of the old code, meaning the `timestamp_granularities` fix hasn't actually executed. Ensure `docker compose build celery-worker && docker compose up -d --force-recreate celery-worker` is run on the server after a `git pull`.
2. **RunPod response format:** RunPod might return the JSON response in a slightly different structure than OpenAI/Groq (e.g., missing `avg_logprob` or structuring `segments` differently), causing a `KeyError` or `AttributeError` in `vad_processor.py` during parsing.
3. **Label Studio:** The pipeline relies on Label Studio project ID `1`. If the LS token is invalid, or project `1` does not exist/accept audio clips, the LS push step will fail.
