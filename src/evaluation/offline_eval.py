import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from src.retrieval.reranker import DocumentReranker


@dataclass(frozen=True)
class OfflineRagReport:
    sample_count: int
    hit_at_1: float
    mean_reciprocal_rank: float
    citation_metadata_accuracy: float

    def passed(self, threshold: float = 0.8) -> bool:
        return (
            min(
                self.hit_at_1,
                self.mean_reciprocal_rank,
                self.citation_metadata_accuracy,
            )
            >= threshold
        )


async def evaluate_dataset(path: Path) -> OfflineRagReport:
    """Check reranking of supplied passages; this does not run document QA."""
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("RAG evaluation dataset is empty.")

    reranker = DocumentReranker()
    hits = 0
    reciprocal_rank = 0.0
    valid_citations = 0

    for sample in samples:
        documents = [
            Document(
                page_content=passage["text"],
                metadata={
                    "id": passage["id"],
                    "filename": passage["filename"],
                    "page": passage["page"],
                },
            )
            for passage in sample["passages"]
        ]
        ranked = await reranker.rerank(sample["question"], documents, top_k=len(documents))
        ranked_ids = [document.metadata["id"] for document in ranked]
        relevant_id = sample["relevant_id"]
        rank = ranked_ids.index(relevant_id) + 1
        hits += int(rank == 1)
        reciprocal_rank += 1 / rank

        relevant = next(document for document in ranked if document.metadata["id"] == relevant_id)
        valid_citations += int(
            bool(relevant.metadata.get("filename"))
            and isinstance(relevant.metadata.get("page"), int)
        )

    count = len(samples)
    return OfflineRagReport(
        sample_count=count,
        hit_at_1=hits / count,
        mean_reciprocal_rank=reciprocal_rank / count,
        citation_metadata_accuracy=valid_citations / count,
    )
