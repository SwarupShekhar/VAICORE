from transcriber import process_transcription_job

job_id = "4f2c6076-21b2-4d63-a851-67d3b274202c"
file_path = "uploads/4f2c6076-21b2-4d63-a851-67d3b274202c_7603700489753002529_1_7603700489753002530.mp4"

print("Starting manual test...")
process_transcription_job(job_id, file_path)
print("Manual test finished.")
