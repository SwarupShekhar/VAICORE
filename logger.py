import os
import sys
import logging
from pythonjsonlogger import jsonlogger

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    
    # Remove existing handlers
    while logger.handlers:
        logger.handlers.pop()
        
    handler = logging.StreamHandler(sys.stdout)
    
    # Custom format
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"}
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Get LOG_LEVEL from env
    env_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    env_level = getattr(logging, env_level_str, logging.INFO)
    
    # Custom rule for vaidikai.processor to hardcap at INFO (i.e. take higher of env_level and INFO)
    if name.startswith("vaidikai.processor"):
        resolved_level = max(env_level, logging.INFO)
    else:
        resolved_level = env_level
        
    logger.setLevel(resolved_level)
    handler.setLevel(resolved_level)
    
    return logger
