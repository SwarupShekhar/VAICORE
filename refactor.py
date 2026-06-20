import os

REPLACEMENTS = {
    "bajaj_processor": "vad_processor",
    "process_bajaj": "process_vad",
    "task_run_bajaj_pipeline": "task_run_vad_pipeline",
    "run_bajaj_pipeline": "run_vad_pipeline",
    "export_bajaj_vad": "export_vad",
    "LABEL_STUDIO_BAJAJ_PROJECT_ID": "LABEL_STUDIO_VAD_PROJECT_ID",
    "BAJAJ_LS_XML": "VAD_LS_XML",
    "bajaj_clips": "vad_clips",
    'value="bajaj"': 'value="vad"',
    'category == "bajaj"': 'category == "vad"',
    'cat == "bajaj"': 'cat == "vad"',
    "bajaj-whisper-v1": "vad-whisper-v1",
    "BAJAJ VOICE AS DATA": "VOICE AS DATA",
    "BAJAJ_UNKNOWN_THRESHOLD": "VAD_UNKNOWN_THRESHOLD",
    "BAJAJ_LANGUAGE": "VAD_LANGUAGE",
    "_BAJAJ_PROJECT_ID": "_VAD_PROJECT_ID",
    "vaidikai_bajaj": "vaidikai_vad",
    "Bajaj Finance Voice as Data pipeline": "Voice as Data pipeline"
}

FILES_TO_PROCESS = [
    "main.py",
    "tasks.py",
    "export_handler.py",
    ".env",
    "vad_processor.py",
    "static/client_upload.html",
    "static/dashboard.html"
]

def refactor():
    for filepath in FILES_TO_PROCESS:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        for old_str, new_str in REPLACEMENTS.items():
            content = content.replace(old_str, new_str)
            
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Updated {filepath}")

if __name__ == "__main__":
    refactor()
