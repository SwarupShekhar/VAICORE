from redis import Redis
from rq import Queue
from transcriber import process_transcription_job

job_id = "0fca244b-53e2-4fc0-b0a3-346e7b097a7c" # ID from screenshot
file_path = "uploads/1c03eb47-109a-4a34-9120-5d86076bf320_6bca4a51-24db-4d35-86e8-c80f88a8639c.mp3" # Using the MP3 we saw earlier or MP4

redis_conn = Redis()
q = Queue(connection=redis_conn)
q.enqueue(process_transcription_job, "b361d5b8-aecb-48b5-a42a-ebe5d19df95c", "uploads/4f2c6076-21b2-4d63-a851-67d3b274202c_7603700489753002529_1_7603700489753002530.mp4", job_timeout='1h')
print("Job enqueued.")
