import stable_whisper
print("Loading model...")
model = stable_whisper.load_model("base")
print("Model loaded.")

def my_progress(seek, total):
    print(f"Progress: {seek} / {total}")

model.transcribe("uploads/4f2c6076-21b2-4d63-a851-67d3b274202c_7603700489753002529_1_7603700489753002530.mp4", progress_callback=my_progress)
