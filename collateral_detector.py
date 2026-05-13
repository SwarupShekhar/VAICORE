"""
Duplicate Collateral Detection Engine
======================================
Zero-infrastructure, OpenCV-only implementation that generates a triple-signature
fingerprint for each annotated gold item and checks it against historical records
to detect "Asset Recycling" (loan stacking) fraud.

Signature Components:
    1. Perceptual Hash (pHash) - 64-bit visual fingerprint
    2. Color Histogram       - HSV color distribution of the gold region
    3. Hu Moments            - 7 rotation-invariant shape descriptors
"""

import json
import os
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np


# ── Configuration ───────────────────────────────────────────────
SIGNATURES_DB_PATH = os.path.join(os.path.dirname(__file__), "collateral_signatures.json")
SIMILARITY_THRESHOLD = float(os.getenv("COLLATERAL_SIMILARITY_THRESHOLD", "0.90"))

# Weights for combined similarity score
W_PHASH = 0.50      # Visual appearance is the strongest signal
W_HISTOGRAM = 0.30   # Color distribution catches material differences
W_HU_MOMENTS = 0.20  # Shape geometry catches structural differences


# ── Core Signature Generation ───────────────────────────────────

def _compute_phash(image: np.ndarray, hash_size: int = 8) -> str:
    """
    Compute a perceptual hash (pHash) of an image.
    Resizes to 32x32, applies DCT, takes top-left 8x8 block,
    and creates a binary hash based on median value.
    Returns a hex string.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    binary = (dct_low > median).flatten()
    # Pack bits into hex
    hash_int = 0
    for bit in binary:
        hash_int = (hash_int << 1) | int(bit)
    return format(hash_int, f'0{hash_size * hash_size // 4}x')


def _compute_color_histogram(image: np.ndarray) -> List[float]:
    """
    Compute a normalized HSV color histogram for the cropped region.
    Uses H (18 bins) and S (8 bins) channels to capture gold tone variations.
    Returns a flat list of 144 normalized values.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()


def _compute_hu_moments(contour_points: np.ndarray) -> List[float]:
    """
    Compute 7 Hu Moments from a contour.
    Log-transforms the values for better numerical comparison.
    """
    moments = cv2.moments(contour_points)
    hu = cv2.HuMoments(moments).flatten()
    # Log-transform: sign(h) * log10(|h|) — handles zero/negative values
    log_hu = []
    for h in hu:
        if abs(h) > 1e-10:
            log_hu.append(-1 * math.copysign(1, h) * math.log10(abs(h)))
        else:
            log_hu.append(0.0)
    return log_hu


def generate_signature(
    image: np.ndarray,
    polygon_points: List[List[float]],
    category: str,
    client_code: str,
    task_id: int,
    item_index: int,
    image_file: str,
    img_width: int,
    img_height: int,
) -> Dict[str, Any]:
    """
    Generate a triple-signature fingerprint for a single annotated jewelry item.
    
    Args:
        image:          The full original image (BGR, as read by cv2)
        polygon_points: List of [x%, y%] points from Label Studio (percentage coordinates)
        category:       e.g. "Ring", "Necklace", "Earring"
        client_code:    e.g. "HDFC_Branch_Mumbai"
        task_id:        Label Studio task ID
        item_index:     Item number within the task
        image_file:     Original filename
        img_width:      Image width in pixels
        img_height:     Image height in pixels
    
    Returns:
        A signature dictionary ready for storage and comparison.
    """
    # Convert percentage points to pixel coordinates
    pixel_points = np.array([
        [int(p[0] * img_width / 100.0), int(p[1] * img_height / 100.0)]
        for p in polygon_points
    ], dtype=np.int32)
    
    # Crop using bounding rect
    x, y, w, h = cv2.boundingRect(pixel_points)
    # Clamp to image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_width - x)
    h = min(h, img_height - y)
    
    if w < 5 or h < 5:
        # Too small to generate meaningful signature
        return None
    
    crop = image[y:y+h, x:x+w]
    
    # Generate all three signatures
    phash = _compute_phash(crop)
    histogram = _compute_color_histogram(crop)
    hu_moments = _compute_hu_moments(pixel_points.reshape((-1, 1, 2)))
    
    return {
        "task_id": task_id,
        "client_code": client_code,
        "category": category,
        "item_index": item_index,
        "phash": phash,
        "hu_moments": hu_moments,
        "color_histogram": histogram,
        "timestamp": datetime.now().isoformat(),
        "image_file": image_file,
    }


# ── Similarity Computation ──────────────────────────────────────

def _hamming_distance(hash1: str, hash2: str) -> int:
    """Count the number of differing bits between two hex hash strings."""
    if len(hash1) != len(hash2):
        return 64  # Maximum distance
    val1 = int(hash1, 16)
    val2 = int(hash2, 16)
    xor = val1 ^ val2
    return bin(xor).count('1')


def _phash_similarity(hash1: str, hash2: str) -> float:
    """Convert Hamming distance to a 0-1 similarity score."""
    distance = _hamming_distance(hash1, hash2)
    return 1.0 - (distance / 64.0)


def _histogram_similarity(hist1: List[float], hist2: List[float]) -> float:
    """Compute correlation-based similarity between two color histograms."""
    h1 = np.array(hist1, dtype=np.float32)
    h2 = np.array(hist2, dtype=np.float32)
    if len(h1) != len(h2):
        return 0.0
    result = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    # CORREL returns -1 to 1; normalize to 0-1
    return max(0.0, (result + 1.0) / 2.0)


def _hu_similarity(hu1: List[float], hu2: List[float]) -> float:
    """Compute similarity between two sets of log-transformed Hu moments."""
    if len(hu1) != len(hu2):
        return 0.0
    h1 = np.array(hu1)
    h2 = np.array(hu2)
    distance = np.linalg.norm(h1 - h2)
    # Convert distance to similarity (exponential decay)
    return math.exp(-distance * 0.5)


def compute_similarity(sig1: Dict, sig2: Dict) -> float:
    """
    Compute the weighted combined similarity between two signatures.
    Returns a score from 0.0 (completely different) to 1.0 (identical).
    """
    p_sim = _phash_similarity(sig1["phash"], sig2["phash"])
    h_sim = _histogram_similarity(sig1["color_histogram"], sig2["color_histogram"])
    hu_sim = _hu_similarity(sig1["hu_moments"], sig2["hu_moments"])
    
    combined = (W_PHASH * p_sim) + (W_HISTOGRAM * h_sim) + (W_HU_MOMENTS * hu_sim)
    return round(combined, 4)


# ── Database Operations ─────────────────────────────────────────

def _load_signatures_db(db_path: str = None) -> List[Dict]:
    """Load the signatures database from disk."""
    path = db_path or SIGNATURES_DB_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("signatures", [])
    except (json.JSONDecodeError, KeyError):
        return []


def _save_signatures_db(signatures: List[Dict], db_path: str = None):
    """Persist the signatures database to disk."""
    path = db_path or SIGNATURES_DB_PATH
    with open(path, "w") as f:
        json.dump({"signatures": signatures, "updated_at": datetime.now().isoformat()}, f, indent=2)


def find_duplicates(
    new_signatures: List[Dict],
    threshold: float = None,
    db_path: str = None,
) -> List[Dict[str, Any]]:
    """
    Check a list of new signatures against the historical database.
    
    Args:
        new_signatures: List of signature dicts to check
        threshold:      Minimum similarity score to flag (default from env)
        db_path:        Path to signatures JSON file
    
    Returns:
        List of match records, each containing:
            - new_item: the new signature that triggered the match
            - matched_item: the historical signature it matched against
            - similarity: the combined similarity score
    """
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    
    existing = _load_signatures_db(db_path)
    if not existing:
        return []
    
    matches = []
    for new_sig in new_signatures:
        if new_sig is None:
            continue
        for old_sig in existing:
            # Skip self-matches (same task)
            if old_sig.get("task_id") == new_sig.get("task_id"):
                continue
            
            # Quick pre-filter: only compare same category
            if old_sig.get("category") != new_sig.get("category"):
                continue
            
            similarity = compute_similarity(new_sig, old_sig)
            if similarity >= threshold:
                matches.append({
                    "new_item": {
                        "client_code": new_sig["client_code"],
                        "category": new_sig["category"],
                        "item_index": new_sig["item_index"],
                        "image_file": new_sig["image_file"],
                        "timestamp": new_sig["timestamp"],
                    },
                    "matched_item": {
                        "client_code": old_sig["client_code"],
                        "category": old_sig["category"],
                        "item_index": old_sig["item_index"],
                        "image_file": old_sig["image_file"],
                        "timestamp": old_sig["timestamp"],
                        "task_id": old_sig["task_id"],
                    },
                    "similarity": similarity,
                })
    
    # Sort by highest similarity first
    matches.sort(key=lambda m: m["similarity"], reverse=True)
    return matches


def store_signatures(new_signatures: List[Dict], db_path: str = None):
    """
    Store new signatures into the persistent database.
    Called only after a successful delivery (no duplicates or admin override).
    """
    existing = _load_signatures_db(db_path)
    for sig in new_signatures:
        if sig is not None:
            existing.append(sig)
    _save_signatures_db(existing, db_path)
    print(f"[Collateral Detector] Stored {len(new_signatures)} new signatures. Total in DB: {len(existing)}")
