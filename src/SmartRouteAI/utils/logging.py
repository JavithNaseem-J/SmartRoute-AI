import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if logger.handlers:
        return logger
    
    if format_string is None:
        format_string = (
            "%(asctime)s | %(name)s | %(levelname)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger(
    "cost_optimized_rag",
    log_file="logs/app.log",
    level="INFO"
)