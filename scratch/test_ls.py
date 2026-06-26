import os
import requests
import json
from dotenv import load_dotenv

load_dotenv('/Users/swarupshekhar/VaidikAIClientportal/vaidikai-portal/.env')
ls_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/")
ls_token = os.getenv("LABEL_STUDIO_TOKEN")

headers = {"Authorization": f"Token {ls_token}"}
r = requests.get(f"{ls_url}/api/tasks?project=1&page_size=5", headers=headers)
if r.status_code == 200:
    tasks = r.json()
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]
    if tasks:
        print("Keys:", tasks[0].keys())
        print("total_annotations:", tasks[0].get("total_annotations"))
        print("annotations:", len(tasks[0].get("annotations", [])))
        print("is_labeled:", tasks[0].get("is_labeled"))
    else:
        print("No tasks found in project 1")
else:
    print(r.status_code, r.text)
