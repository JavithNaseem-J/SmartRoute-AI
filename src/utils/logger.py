import os
import sys
import logging
from datetime import datetime


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)


# Custom StreamHandler with UTF-8 encoding for Windows
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Handle Windows encoding issues
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback: replace non-ASCII characters
            msg = self.format(record).encode('ascii', 'replace').decode('ascii')
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath, encoding='utf-8'),
        UTF8StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger("SmartRouteAILogger")
