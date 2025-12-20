import json
import time
from pathlib import Path
from typing import Any, Dict, List
from functools import wraps
import hashlib


def ensure_dir(path: str) -> Path:
    """Ensure directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """Save data to JSON file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def append_jsonl(data: Dict, file_path: str):
    """Append data to JSONL file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'a') as f:
        f.write(json.dumps(data) + '\n')


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    if not Path(file_path).exists():
        return []
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def timer(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


def hash_text(text: str) -> str:
    """Generate hash for text"""
    return hashlib.md5(text.encode()).hexdigest()


def count_words(text: str) -> int:
    """Count words in text"""
    return len(text.split())


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_cost(cost: float) -> str:
    """Format cost as currency"""
    return f"${cost:.4f}"


def format_tokens(tokens: int) -> str:
    """Format token count"""
    if tokens < 1000:
        return f"{tokens}"
    elif tokens < 1_000_000:
        return f"{tokens/1000:.1f}K"
    else:
        return f"{tokens/1_000_000:.1f}M"