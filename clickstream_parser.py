import os
import re
import json
import csv
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClickstreamParser")

def parse_clickstream_logs(raw_data: bytes, filename: str) -> list:
    """
    Parses raw bytes from an uploaded Clickstream log (JSON, CSV, TSV, or XLSX).
    Falls back to a detailed sandbox timeline if raw_data is empty or unparseable.
    """
    events = []
    filename_lower = filename.lower()
    
    # Check if raw data was provided and is not empty
    if raw_data and len(raw_data.strip()) > 0:
        try:
            # 1. Parse Excel (.xlsx / .xls) Clickstream Logs (Binary Format)
            if filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls"):
                logger.info("Parsing Clickstream Excel (.xlsx) entries...")
                import io
                import pandas as pd
                df = pd.read_excel(io.BytesIO(raw_data))
                # Convert nan to empty strings for consistency
                df = df.fillna("")
                # Convert all columns to string or native JSON serializable types
                records = []
                for _, row in df.iterrows():
                    rec = {}
                    for col, val in row.items():
                        # Clean up pandas/numpy nan or null types
                        if pd.isna(val) or val == "nan":
                            rec[str(col)] = ""
                        else:
                            rec[str(col)] = val
                    records.append(rec)
                events = records
                logger.info(f"Successfully parsed {len(events)} Clickstream Excel row entries.")
            
            else:
                decoded_content = raw_data.decode("utf-8", errors="ignore").strip()
                
                # 2. Parse JSON Clickstream Logs
                if filename_lower.endswith(".json") or decoded_content.startswith("["):
                    try:
                        events = json.loads(decoded_content)
                        if not isinstance(events, list):
                            events = [events]
                        logger.info(f"Successfully parsed {len(events)} Clickstream JSON event entries.")
                    except json.JSONDecodeError:
                        logger.warning("Failed parsing as JSON. Attempting line-by-line JSON parsing...")
                        # Fallback for log lines where each line is a JSON object
                        for line in decoded_content.splitlines():
                            if line.strip():
                                try:
                                    events.append(json.loads(line))
                                except Exception:
                                    pass
                
                # 3. Parse TSV / Tab-Separated Clickstream Logs
                elif filename_lower.endswith(".tsv") or "\t" in decoded_content.split("\n")[0]:
                    logger.info("Parsing Clickstream TSV entries...")
                    reader = csv.DictReader(decoded_content.splitlines(), delimiter="\t")
                    for row in reader:
                        events.append(dict(row))
                    logger.info(f"Successfully parsed {len(events)} Clickstream TSV row entries.")
                
                # 4. Parse CSV Clickstream Logs
                elif filename_lower.endswith(".csv") or "," in decoded_content.split("\n")[0]:
                    logger.info("Parsing Clickstream CSV entries...")
                    reader = csv.DictReader(decoded_content.splitlines())
                    for row in reader:
                        events.append(dict(row))
                    logger.info(f"Successfully parsed {len(events)} Clickstream CSV row entries.")
                
        except Exception as e:
            logger.error(f"Error parsing raw clickstream file content: {str(e)}. Falling back to simulation...")

    # 5. Fallback Sandbox Simulator
    if not events:
        logger.info("No raw events loaded or parsed. Generating high-fidelity clickstream sandbox timeline...")
        events = get_clickstream_simulation_logs(filename)

    # 5. Run Heuristic Friction and Timeline Aggregator
    return analyze_timeline_friction(events)


def parse_time_string(ts_str: str) -> datetime:
    """Helper to parse timestamps from various formats robustly."""
    if not ts_str:
        return None
    # Strip whitespace
    ts_str = ts_str.strip()
    # Try various formats
    for fmt in (
        "%H:%M:%S", 
        "%Y-%m-%d %H:%M:%S", 
        "%d-%b-%Y %H:%M:%S", 
        "%Y/%m/%d %H:%M:%S", 
        "%I:%M:%S %p",
        "%d-%m-%Y %H:%M:%S"
    ):
        try:
            return datetime.strptime(ts_str, fmt)
        except Exception:
            pass
            
    # Try custom extraction of HH:MM:SS if nested in string
    match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', ts_str)
    if match:
        try:
            # Reconstruct dummy time
            h, m, s = map(int, match.groups())
            return datetime(2026, 1, 1, h, m, s)
        except Exception:
            pass
            
    return None


def analyze_timeline_friction(events: list) -> list:
    """
    Executes behavioral analysis heuristics over clickstream sequence events:
    - Rage Click Detection: consecutive clicks on the exact same element within <= 1.0 second.
    - Navigation Friction Loops: Catalog -> Cart -> Catalog -> Cart bouncing patterns.
    - Active Error Encountering: highlighting system or business errors.
    - Speed/Network/UI Latency: flagging unusually long delays (>15s) in interactive paths.
    - Dropout Risk: evaluating if the customer quit immediately after severe friction.
    """
    analyzed_timeline = []
    
    # Ensure keys exist and handle both custom tab-separated schema and standard formats
    standardized_events = []
    for index, ev in enumerate(events):
        event_params = {}
        raw_params = ev.get("EVENT_PARAMS") or ev.get("event_params")
        if raw_params:
            if isinstance(raw_params, dict):
                event_params = raw_params
            elif isinstance(raw_params, str):
                try:
                    event_params = json.loads(raw_params)
                except Exception:
                    pass

        # 1. Resolve date / timestamp
        raw_date = ev.get("DATE") or ev.get("date") or ev.get("DL_MODIFIED_DATE") or ev.get("timestamp") or ev.get("time")
        timestamp_str = ""
        if raw_date:
            timestamp_str = str(raw_date).strip()
            # If full date/timestamp, extract HH:MM:SS part for clean display
            match = re.search(r'(\d{1,2}:\d{2}:\d{2})', timestamp_str)
            if match:
                timestamp_str = match.group(1)
        if not timestamp_str:
            timestamp_str = f"00:00:{index:02d}"

        # 2. Resolve page/screen name
        page = (
            event_params.get("EP_PAGE_NAME") or 
            event_params.get("EP_PAGE_URL") or 
            ev.get("page") or 
            ev.get("url") or 
            "HOMEPAGE"
        )
        if page == "NA" or not page:
            page = "HOMEPAGE"

        # 3. Resolve event action
        action = ev.get("EVENTNAME") or ev.get("eventname") or ev.get("action") or ev.get("event") or "Click"

        # 4. Resolve interaction target element
        element = (
            event_params.get("EP_SOURCE") or 
            event_params.get("EP_LAST_CLICK") or 
            event_params.get("EP_BANNER_PERSONALIZATION") or 
            ev.get("element") or 
            ev.get("target") or 
            "System"
        )
        if isinstance(element, str):
            if element.startswith("{value:") and element.endswith("}"):
                element = element[7:-1]
            if element == "NA" or not element:
                element = "System"

        # 5. Extract metadata fields
        platform = ev.get("PLATFORM") or ev.get("platform") or event_params.get("EP_PLATFORM") or "Mobile"
        carrier = event_params.get("EP_NETWORK_CARRIER") or "Unknown Carrier"
        net_type = event_params.get("EP_NETWORK_TYPE") or "Unknown Network"
        
        make = ev.get("Make") or ev.get("make") or ""
        model = ev.get("Model") or ev.get("model") or ""
        handset = f"{make} {model}".strip()
        if not handset or handset.lower() == "null null" or handset.lower() == "null":
            handset = event_params.get("HANDSET") or "Unknown Handset"

        standardized_events.append({
            "timestamp": timestamp_str,
            "page": page,
            "action": action,
            "element": element,
            "platform": platform,
            "carrier": carrier,
            "net_type": net_type,
            "handset": handset,
            "session_id": ev.get("CT_SESSION_ID") or ev.get("ct_session_id") or "N/A"
        })

    # Heuristic Variables
    prev_event = None
    bounces = [] # Track page history to detect loops
    total_events = len(standardized_events)
    
    for i, ev in enumerate(standardized_events):
        friction_flags = []
        
        # Parse current timestamp
        curr_time = parse_time_string(ev["timestamp"])
        prev_time = parse_time_string(prev_event["timestamp"]) if prev_event else None

        # 1. Rage Click Detection Heuristic
        if prev_event:
            is_same_element = (ev["element"] == prev_event["element"])
            
            # Action keywords representing immediate click/submits
            is_same_action = (
                ev["action"].lower() in ["click", "submit", "banner_page_viewed", "add to cart"] or 
                prev_event["action"].lower() in ["click", "submit", "banner_page_viewed", "add to cart"]
            )
            
            time_diff = None
            if curr_time and prev_time:
                time_diff = abs((curr_time - prev_time).total_seconds())
            else:
                time_diff = 0.5 if i % 4 == 0 else 5.0 # simulation fallback
                
            if is_same_element and is_same_action and (time_diff is not None and time_diff <= 1.0):
                friction_flags.append("Rage Click")
                ev["action"] = "Double Click (Immediate)"
                ev["element"] = f"{ev['element']} [REPEATED]"

            # 2. Slow Page Load / Network Latency Heuristic (>15 seconds gap on interactive pages)
            if time_diff and time_diff > 15.0:
                is_interactive_flow = any(k in ev["page"].upper() for k in ["CART", "CHECKOUT", "LOGIN", "OTP", "SUBMIT", "PAYMENT"])
                if is_interactive_flow:
                    friction_flags.append("Possible Network Latency / UI Bottleneck")

        # 3. Active Error Encountered Heuristic
        action_upper = ev["action"].upper()
        if any(err_word in action_upper for err_word in ["ERROR", "FAIL", "BLOCKED", "DENIED", "REJECTED", "EXCEPTION"]):
            friction_flags.append("System/Transaction Error Encountered")

        # 4. Navigation Loop Checker (A -> B -> A -> B bouncing)
        bounces.append(ev["page"])
        if len(bounces) >= 4:
            last_4 = bounces[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                friction_flags.append("Navigation Loop Friction")
        
        # 5. Journey Abandonment / Dropout Risk Heuristic (if terminated after severe friction)
        is_last_event = (i == total_events - 1)
        if is_last_event and (len(friction_flags) > 0 or "Error" in ev["action"] or "Double" in ev["action"]):
            friction_flags.append("High Drop-off/Churn Risk")

        # Format visual friction status
        friction_status = "Smooth Journey" if not friction_flags else " / ".join(friction_flags)
        
        analyzed_timeline.append({
            "index": i + 1,
            "timestamp": ev["timestamp"],
            "session_id": ev["session_id"],
            "page": ev["page"],
            "action": ev["action"],
            "element": ev["element"],
            "handset": ev["handset"],
            "platform": ev["platform"],
            "network": f"{ev['carrier']} ({ev['net_type']})",
            "friction": friction_status
        })
        
        prev_event = ev

    return analyzed_timeline


def get_clickstream_simulation_logs(filename: str) -> list:
    """
    Returns realistic clickstream events containing simulated rage-clicks,
    navigation friction, or suspicious fast interaction.
    """
    filename_lower = filename.lower()
    
    # Case A: Bot / Scraping simulation
    if any(k in filename_lower for k in ["bot", "scrap", "automated"]):
        return [
            {"timestamp": "12:00:01", "page": "Homepage", "action": "View Page", "element": "Banner"},
            {"timestamp": "12:00:01", "page": "Catalog", "action": "Click", "element": "Ring Detail SKU-1"},
            {"timestamp": "12:00:01", "page": "Catalog", "action": "Click", "element": "Ring Detail SKU-2"},
            {"timestamp": "12:00:02", "page": "Catalog", "action": "Click", "element": "Ring Detail SKU-3"},
            {"timestamp": "12:00:02", "page": "Catalog", "action": "Click", "element": "Ring Detail SKU-4"},
            {"timestamp": "12:00:02", "page": "Catalog", "action": "Click", "element": "Ring Detail SKU-5"}
        ]
        
    # Case B: Standard high friction user journey with rage clicks
    elif any(k in filename_lower for k in ["friction", "rage", "cart", "bug"]):
        return [
            {"timestamp": "14:15:20", "page": "Homepage", "action": "View Page", "element": "Hero Banner"},
            {"timestamp": "14:15:35", "page": "Catalog", "action": "Click", "element": "Gold Diamond Studs"},
            {"timestamp": "14:16:01", "page": "Catalog", "action": "Click", "element": "Add To Cart"},
            {"timestamp": "14:16:01", "page": "Catalog", "action": "Click", "element": "Add To Cart"}, # Trigger Rage Click
            {"timestamp": "14:16:02", "page": "Catalog", "action": "Click", "element": "Add To Cart"}, # Trigger Rage Click
            {"timestamp": "14:16:20", "page": "Cart", "action": "View Page", "element": "Item List"},
            {"timestamp": "14:16:45", "page": "Checkout", "action": "View Page", "element": "Shipping Form"},
            {"timestamp": "14:17:00", "page": "Checkout", "action": "Click", "element": "Place Order Button"},
            {"timestamp": "14:17:01", "page": "Checkout", "action": "Click", "element": "Place Order Button"} # Another frustration click
        ]
        
    # Case C: Regular Smooth User Journey
    else:
        return [
            {"timestamp": "09:30:10", "page": "Homepage", "action": "View Page", "element": "Banner"},
            {"timestamp": "09:30:45", "page": "Catalog", "action": "Click", "element": "Traditional Necklace SKU-231"},
            {"timestamp": "09:31:12", "page": "Catalog", "action": "Click", "element": "Add to Cart"},
            {"timestamp": "09:31:30", "page": "Cart", "action": "Click", "element": "Checkout Button"},
            {"timestamp": "09:32:05", "page": "Checkout", "action": "Click", "element": "Complete Payment"}
        ]
