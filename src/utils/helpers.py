import json
from pathlib import Path
from typing import Any, Dict


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


def save_json(data: Dict, filepath: Path):
    """Save data to JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: Path) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def append_jsonl(data: Dict, filepath: Path):
    """Append data to JSONL file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(json.dumps(data) + '\n')


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def calculate_percentage(part: float, total: float) -> float:
    """Calculate percentage safely"""
    if total == 0:
        return 0.0
    return (part / total) * 100