from fastapi import APIRouter
from shared.models.common import RetrievalQuery, RetrievalResponse

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrievalResponse, status_code=200)
async def retrieve_chunks(query: RetrievalQuery) -> RetrievalResponse:
    """
    Scaffolding endpoint for chunk retrieval.
    TODO: Implement hybrid search querying Qdrant (dense vectors) and
    Elasticsearch (BM25 keywords), with score normalization and reranking.
    """
    return RetrievalResponse(
        results=[],
        total_found=0,
    )
