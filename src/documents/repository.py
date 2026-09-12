from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.cost.tracker import DocumentAsset


def _serialize(document: DocumentAsset) -> Dict[str, Any]:
    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "storage_bucket": document.storage_bucket,
        "storage_path": document.storage_path,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "modified_time": document.created_at.timestamp() if document.created_at else None,
    }


def create_document_record(
    tracker,
    *,
    user_id: str,
    filename: str,
    content_type: str,
    size_bytes: int,
    storage_bucket: str,
    storage_path: str,
) -> Dict[str, Any]:
    document = DocumentAsset(
        id=str(uuid4()),
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        storage_bucket=storage_bucket,
        storage_path=storage_path,
    )
    with tracker._get_session() as session:
        session.add(document)
        session.commit()
        session.refresh(document)
        return _serialize(document)


def list_active_documents(tracker, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with tracker._get_session() as session:
        query = session.query(DocumentAsset).filter(DocumentAsset.deleted_at.is_(None))
        if user_id:
            query = query.filter(DocumentAsset.user_id == user_id)
        documents = query.order_by(DocumentAsset.created_at.desc()).all()
        return [_serialize(document) for document in documents]


def get_active_document(
    tracker, filename: str, user_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    with tracker._get_session() as session:
        query = session.query(DocumentAsset).filter(
            DocumentAsset.filename == filename,
            DocumentAsset.deleted_at.is_(None),
        )
        if user_id:
            query = query.filter(DocumentAsset.user_id == user_id)
        document = query.order_by(DocumentAsset.created_at.desc()).first()
        return _serialize(document) if document else None


def mark_document_deleted(tracker, storage_path: str) -> None:
    with tracker._get_session() as session:
        document = (
            session.query(DocumentAsset)
            .filter(
                DocumentAsset.storage_path == storage_path,
                DocumentAsset.deleted_at.is_(None),
            )
            .first()
        )
        if document:
            document.deleted_at = datetime.utcnow()
            session.commit()
