from typing import Optional, TypedDict


class Citation(TypedDict):
    """Structured evidence returned with a grounded answer."""

    id: str
    filename: str
    page: Optional[int]
    section: Optional[str]
    excerpt: str
