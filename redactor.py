import re
import logging
from typing import List, Dict, Any
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Redactor")

# --- REGEX PROFILES FOR SENSITIVE DATA ---

# Aadhaar Cards: 12 digits (with optional spaces: e.g. 1234 5678 9012 or 1234-5678-9012 or continuous 12 digits)
AADHAAR_REGEX = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")

# PAN Cards: 5 uppercase letters, 4 digits, 1 uppercase letter (e.g., ABCDE1234F)
PAN_REGEX = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)

# Indian Phone Numbers: +91, 91, or 0 followed by 10 digits starting with 6-9
PHONE_REGEX = re.compile(r"\b(?:\+?91[-\s]?)?[6-9]\d{9}\b|\b(?:\+?91[-\s]?)?[6-9]\d{4}[-\s]?\d{5}\b")

# Bank Account Numbers: typically between 9 and 18 digits inside financial contexts
BANK_ACCOUNT_REGEX = re.compile(r"\b\d{9,18}\b")

# GSTIN (Goods and Services Tax Identification Number): 15-character alphanumeric (e.g. 21BUAPB0758D1ZH)
GSTIN_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]{3}\b", re.IGNORECASE)

# Common name-indicators to aid regex-based name detection in standard document outputs
NAME_PATTERNS = [
    re.compile(r"Name\s*:\s*([A-Za-z\s]{2,30})(?=\n|\.)", re.IGNORECASE),
    re.compile(r"Customer\s+Name\s*:\s*([A-Za-z\s]{2,30})(?=\n|\.)", re.IGNORECASE),
    re.compile(r"Reviewer\s*:\s*([A-Za-z\s]{2,30})(?=\n|\.)", re.IGNORECASE)
]

def mask_text_data(text: str) -> str:
    """
    Masks Names, Phone Numbers, Bank Accounts, Aadhaar Cards, and PAN Cards inside raw text logs.
    Executes entirely locally to maintain absolute privacy.
    """
    if not text:
        return ""
    
    redacted = text
    
    # 1. Mask Aadhaar Card Numbers
    redacted = AADHAAR_REGEX.sub("[REDACTED_AADHAAR]", redacted)
    
    # 2. Mask GSTIN Numbers
    redacted = GSTIN_REGEX.sub("[REDACTED_FINANCIAL_ID]", redacted)
    
    # 3. Mask PAN Card Numbers
    redacted = PAN_REGEX.sub("[REDACTED_PAN]", redacted)
    
    # 3. Mask Phone Numbers
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", redacted)
    
    # 4. Mask Explicit Name Headers (Name: John Doe -> Name: [REDACTED_NAME])
    for pattern in NAME_PATTERNS:
        match = pattern.search(redacted)
        if match:
            name = match.group(1).strip()
            # Ensure we don't accidentally mask common fields like "Phone" or "Aadhaar" if matched
            if len(name) > 1 and "Phone" not in name and "Account" not in name:
                redacted = redacted.replace(name, "[REDACTED_NAME]")
                
    # 5. Mask Bank Accounts (by identifying digit sequences adjacent to "account", "acc", "bank" etc.)
    # We do a contextual search first to avoid masking general numbers (like dates, invoices)
    financial_keywords = ["account", "acc", "bank", "invoice no", "card number"]
    words = redacted.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in financial_keywords:
            # Look at subsequent words to find numbers and mask them
            for j in range(i + 1, min(i + 4, len(words))):
                sub_word = words[j]
                if BANK_ACCOUNT_REGEX.match(sub_word):
                    words[j] = "[REDACTED_FINANCIAL_ID]"
    
    redacted = " ".join(words)
    
    logger.info("PII redaction scrubbing executed successfully.")
    return redacted


def redact_faces_in_image(image: np.ndarray, segments: List[Dict[str, Any]]) -> np.ndarray:
    """
    Apply Gaussian blur to regions matching 'borrower_person' or 'representative_person'
    to anonymize and redact face PII prior to data exports.
    """
    import cv2
    if image is None or not segments:
        return image
        
    h, w = image.shape[:2]
    redacted_image = image.copy()
    
    for item in segments:
        cat = str(item.get("Category", "")).lower()
        if "person" in cat or "borrower" in cat or "representative" in cat:
            pts = item.get("Points", [])
            if not pts:
                continue
                
            # Convert percentage points to pixel coordinates
            pixel_points = np.array([
                [int(p[0] * w / 100.0), int(p[1] * h / 100.0)]
                for p in pts
            ], dtype=np.int32)
            
            # Get bounding rect for blur
            rx, ry, rw, rh = cv2.boundingRect(pixel_points)
            
            # Clamp to image boundaries
            rx1 = max(0, rx)
            ry1 = max(0, ry)
            rx2 = min(w, rx + rw)
            ry2 = min(h, ry + rh)
            
            if (rx2 - rx1) > 5 and (ry2 - ry1) > 5:
                # Apply high-sigma Gaussian blur
                roi = redacted_image[ry1:ry2, rx1:rx2]
                # Calculate kernel size based on ROI size (must be odd)
                k_w = int((rx2 - rx1) | 1) # Force odd number
                k_h = int((ry2 - ry1) | 1) # Force odd number
                # Cap the maximum kernel size to avoid slow computation, but keep it blurry
                k_w = min(99, max(15, k_w))
                k_h = min(99, max(15, k_h))
                
                blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 30)
                redacted_image[ry1:ry2, rx1:rx2] = blurred_roi
                
    return redacted_image

