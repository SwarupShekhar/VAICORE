import os
import time
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RunPodClient")

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "mock_endpoint")

def run_runpod_inference(file_url: str, task_type: str = "jewelry") -> dict:
    """
    Submits a task to RunPod Serverless GPU Endpoint and polls for completion.
    If RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID is missing or configured as a mock,
    runs an instant local prediction simulation.
    """
    # 1. Check for Offline/Mock Mode
    if not RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID == "mock_endpoint" or RUNPOD_API_KEY.startswith("mock"):
        logger.info(f"[Mock Mode] Simulating local GPU inference for: {file_url} (Type: {task_type})")
        time.sleep(1.0) # Simulate network/GPU latency

        if task_type == "jewelry":
            # Hardcoded "test image" polygons were removed: they pasted fixed
            # coordinates tuned for ONE photo onto every upload, which is why
            # preannotations landed on empty tray felt. Every image now runs
            # real pixel-based detection so regions track the actual photo.
            predictions = []
            try:
                import cv2
                import numpy as np

                # Resolve local path if possible (avoids a re-download)
                if "api/files/" in file_url:
                    parts = file_url.split("api/files/")[-1].split("/")
                    client_code = parts[0]
                    filename = "/".join(parts[1:])
                    local_path = os.path.join("/tmp/vaidikai", client_code, filename)
                else:
                    local_path = None

                img = None
                if local_path and os.path.exists(local_path):
                    img = cv2.imread(local_path)
                else:
                    logger.info(f"Downloading image from {file_url} for OpenCV detection.")
                    resp = requests.get(file_url, timeout=10)
                    if resp.status_code == 200:
                        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    height, width = img.shape[:2]
                    img_area = float(width * height) or 1.0
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                    # (a) Warm gold by hue. Saturation/value floors at 50 keep
                    #     dark felt out while still catching matte gold.
                    lower_gold = np.array([8, 50, 50], dtype=np.uint8)
                    upper_gold = np.array([35, 255, 255], dtype=np.uint8)
                    gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)

                    # (b) Bright specular sparkle on polished gold / thin chains
                    #     reads as low-saturation bright pixels. Keep only the
                    #     warm ones (R clearly above B) so grey felt glare and
                    #     pure-white highlights are excluded.
                    bright = cv2.inRange(hsv[:, :, 2], 205, 255)
                    b_ch, g_ch, r_ch = cv2.split(img.astype(np.int16))
                    warm = ((r_ch - b_ch) > 18).astype(np.uint8) * 255
                    bright_gold = cv2.bitwise_and(bright, warm)

                    mask = cv2.bitwise_or(gold_mask, bright_gold)

                    # OPEN first to snap the thin felt bridges that fuse
                    # separate pieces into one giant blob, then a small CLOSE
                    # to fill holes inside a single piece.
                    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Silver / platinum fallback: if almost no gold, use Otsu.
                    gold_total = sum(cv2.contourArea(c) for c in contours)
                    if not contours or gold_total < img_area * 0.001:
                        logger.info("[OpenCV] No gold detected. Falling back to grayscale Otsu.")
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
                        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open, iterations=1)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    def _classify(cnt):
                        area = cv2.contourArea(cnt)
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        bbox_area = float(bw * bh) or 1.0
                        extent = area / bbox_area
                        long_side = float(max(bw, bh))
                        short_side = float(min(bw, bh)) or 1.0
                        aspect = long_side / short_side
                        hull_area = cv2.contourArea(cv2.convexHull(cnt)) or 1.0
                        solidity = area / hull_area
                        rel = area / img_area

                        # Long & thin -> a chain strand
                        if aspect >= 3.0:
                            return "Chain"
                        # Big, open / wiry loop -> necklace
                        if rel > 0.025 and (solidity < 0.6 or aspect >= 1.8):
                            return "Necklace"
                        # Small, compact, near-round -> ring
                        if rel < 0.012 and extent >= 0.5 and aspect < 1.7:
                            return "Ring"
                        # Medium compact cluster -> earrings
                        if rel < 0.06:
                            return "Earrings"
                        return "General Jewelry"

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        rel = area / img_area
                        # Drop dust / specks and the whole-tray felt blob.
                        if rel < 0.0015 or rel > 0.45:
                            continue
                        epsilon = 0.005 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        if len(approx) < 3:
                            continue
                        points = [[(p[0][0] / width) * 100.0,
                                   (p[0][1] / height) * 100.0] for p in approx]
                        predictions.append({
                            "class": _classify(cnt),
                            "points": points,
                            "confidence": 0.80
                        })
                    logger.info(f"[OpenCV] Detected {len(predictions)} jewelry regions from actual pixels.")
            except Exception as e:
                logger.error(f"OpenCV detection failed: {e}")

            # Ultimate failsafe so the annotator always gets at least one
            # region to adjust rather than an empty task.
            if not predictions:
                predictions = [
                    {
                        "class": "General Jewelry",
                        "points": [[20.0, 25.0], [35.0, 25.0], [35.0, 40.0], [20.0, 40.0]],
                        "confidence": 0.50
                    }
                ]

            counts = {}
            for p in predictions:
                c = p["class"]
                counts[c] = counts.get(c, 0) + 1
            counts["total"] = len(predictions)

            return {
                "status": "success",
                "engine": "OpenCV-GoldContour-Heuristic",
                "predictions": predictions,
                "counts": counts
            }
        elif task_type == "form":
            return {
                "status": "success",
                "engine": "Mock-EasyOCR",
                "raw_ocr_text": "Customer Invoice. Name: John Doe. Phone: +91 98765 43210. Aadhaar: 1234 5678 9012. PAN: ABCDE1234F. Account No: 9876543210123. Total Payment due: 45,000 INR.",
                "confidence": 0.91
            }
        else:
            return {
                "status": "success",
                "data": "Mock data processed successfully."
            }

    # 2. Real RunPod API Call
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {
            "file_url": file_url,
            "task_type": task_type
        }
    }

    url = f"https://api.runpod.ai/v1/{RUNPOD_ENDPOINT_ID}/run"

    try:
        logger.info(f"Triggering RunPod Serverless Endpoint {RUNPOD_ENDPOINT_ID}...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get("id")

        if not job_id:
            logger.error(f"Failed to obtain job ID from RunPod response: {job_data}")
            return {"status": "error", "message": "No job ID returned"}

        logger.info(f"Job {job_id} successfully queued. Polling for results...")
        status_url = f"https://api.runpod.ai/v1/{RUNPOD_ENDPOINT_ID}/status/{job_id}"

        max_attempts = 60 # 2 minutes max polling
        for attempt in range(max_attempts):
            res_status = requests.get(status_url, headers=headers, timeout=10)
            res_status.raise_for_status()
            res_data = res_status.json()

            status = res_data.get("status")
            if status == "COMPLETED":
                logger.info("RunPod serverless job completed successfully.")
                return {
                    "status": "success",
                    "predictions": res_data.get("output", {}).get("predictions", []),
                    "counts": res_data.get("output", {}).get("counts", {}),
                    "raw_ocr_text": res_data.get("output", {}).get("raw_ocr_text", "")
                }
            elif status in ["FAILED", "CANCELLED"]:
                logger.error(f"RunPod execution failed with status: {status}")
                return {"status": "error", "message": f"RunPod execution {status}"}

            time.sleep(2) # Poll every 2 seconds

        logger.error("RunPod serverless task execution timed out.")
        return {"status": "error", "message": "Execution timeout"}

    except Exception as e:
        logger.error(f"RunPod client API failure: {str(e)}")
        return {"status": "error", "message": str(e)}
