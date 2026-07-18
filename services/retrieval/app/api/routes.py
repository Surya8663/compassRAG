from fastapi import APIRouter

from shared.models.common import RetrievalQuery, RetrievalResult

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrievalResult, status_code=200)
async def retrieve_chunks(query: RetrievalQuery) -> RetrievalResult:
    """
    Scaffolding endpoint for chunk retrieval.
    TODO: Implement hybrid search querying Qdrant (dense vectors) and
    Elasticsearch (BM25 keywords), with score normalization and reranking.
    """
    return RetrievalResult(
        chunks=[],
        total_found=0,
    )
