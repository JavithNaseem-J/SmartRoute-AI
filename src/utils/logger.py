import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# JSON formatter — every log line is a valid JSON object.
# Machine-parseable by ELK, Datadog, CloudWatch, etc.
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

_json_formatter = JsonFormatter()

# Rotating file handler — caps at 10 MB per file, keeps 5 backups (50 MB max).
# Prevents disk exhaustion from an unbounded growing log file.
_file_handler = logging.handlers.RotatingFileHandler(
    log_filepath,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_json_formatter)

# Stdout handler — Docker and Kubernetes collect logs from stdout.
# Use JSON so log aggregators can parse structured fields directly.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setFormatter(_json_formatter)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    handlers=[_file_handler, _stdout_handler],
)

logger = logging.getLogger("SmartRouteAILogger")
