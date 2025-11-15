"""
Utility functions for the AI Document Assistant.
"""
import uuid
from datetime import datetime
from typing import Optional


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp as ISO format string."""
    return datetime.now().isoformat()


def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max_length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def validate_pdf_file(file) -> bool:
    """Validate that uploaded file is a PDF."""
    if file is None:
        return False
    return file.name.lower().endswith('.pdf')


