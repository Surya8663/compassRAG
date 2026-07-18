"""
Embedding Service with Redis cache-aside pattern.
Supports local models (`all-MiniLM-L6-v2`) and OpenAI models (`text-embedding-3-small`).
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any

import redis
from shared.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Real Embedding Service supporting offline and online models.
    Every generation is wrapped in a Redis cache-aside pattern keyed by content hash (`sha256`),
    guaranteeing identical chunk text is never re-embedded.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = self.settings.EMBEDDING_PROVIDER.lower()
        if self.provider == "local":
            self.model_name = self.settings.LOCAL_EMBEDDING_MODEL
            logger.info("Initializing local model: %s", self.model_name)
            from sentence_transformers import SentenceTransformer

            self._local_model: Any = SentenceTransformer(self.model_name)
            self._openai_client: Any = None
            if hasattr(self._local_model, "get_embedding_dimension"):
                self.dimension = self._local_model.get_embedding_dimension() or 384
            else:
                self.dimension = (
                    self._local_model.get_sentence_embedding_dimension() or 384
                )
        elif self.provider == "openai":
            self.model_name = self.settings.OPENAI_EMBEDDING_MODEL
            logger.info("Initializing OpenAI client: %s", self.model_name)
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=self.settings.OPENAI_API_KEY)
            self._local_model = None
            dim = self.settings.EMBEDDING_DIMENSION
            self.dimension = dim if dim != 384 else 1536
        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. Must be 'local' or 'openai'."
            )

        self.redis_client = redis.from_url(
            self.settings.REDIS_URL, decode_responses=True
        )
        self.cache_ttl = 604800  # 7 days in seconds

    def _compute_cache_key(self, content: str) -> str:
        """
        Computes SHA-256 hash of text content for deterministic cache lookup.
        """
        digest = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
        return f"embed_cache:{self.provider}:{self.model_name}:{digest}"

    def embed_text(self, content: str) -> list[float]:
        """
        Embeds a single string into vector representation.
        First checks Redis cache-aside; computes and caches on miss.
        """
        if not content or not isinstance(content, str) or not content.strip():
            return [0.0] * self.dimension

        cache_key = self._compute_cache_key(content)
        try:
            cached = self.redis_client.get(cache_key)
            if cached is not None:
                logger.debug("Redis cache hit for %s", cache_key)
                vector: list[float] = json.loads(cached)
                return vector
        except Exception as exc:
            logger.warning("Redis read error (%s): %s", cache_key, exc)

        logger.debug("Redis miss. Generating embedding with %s...", self.provider)
        vector = self._generate_embedding(content)

        try:
            self.redis_client.set(cache_key, json.dumps(vector), ex=self.cache_ttl)
        except Exception as exc:
            logger.warning("Redis write error (%s): %s", cache_key, exc)

        return vector

    def embed_batch(self, contents: list[str]) -> list[list[float]]:
        """
        Embeds a batch of strings utilizing Redis cache-aside.
        Only cache misses are passed to the model in a single batch inference call.
        """
        if not contents:
            return []

        results: list[list[float] | None] = [None] * len(contents)
        miss_indices: list[int] = []
        miss_contents: list[str] = []
        cache_keys: list[str] = []

        for idx, text in enumerate(contents):
            if not text or not text.strip():
                results[idx] = [0.0] * self.dimension
                continue

            key = self._compute_cache_key(text)
            cache_keys.append(key)
            try:
                cached = self.redis_client.get(key)
                if cached is not None:
                    results[idx] = json.loads(cached)
                    continue
            except Exception as exc:
                logger.warning("Redis batch get error (%s): %s", key, exc)

            miss_indices.append(idx)
            miss_contents.append(text)

        if miss_contents:
            logger.debug(
                "Computing %d misses via %s batch...", len(miss_contents), self.provider
            )
            miss_vectors = self._generate_batch_embeddings(miss_contents)
            for idx_in_miss, orig_idx in enumerate(miss_indices):
                vec = miss_vectors[idx_in_miss]
                results[orig_idx] = vec
                key = self._compute_cache_key(contents[orig_idx])
                try:
                    self.redis_client.set(key, json.dumps(vec), ex=self.cache_ttl)
                except Exception as exc:
                    logger.warning("Redis batch set error (%s): %s", key, exc)

        return [r if r is not None else [0.0] * self.dimension for r in results]

    def _generate_embedding(self, text: str) -> list[float]:
        """
        Directly invokes the underlying model provider without cache lookup.
        """
        if self.provider == "local" and self._local_model is not None:
            raw_vec = self._local_model.encode(text, normalize_embeddings=True)
            return [float(x) for x in raw_vec]
        elif self.provider == "openai" and self._openai_client is not None:
            resp = self._openai_client.embeddings.create(
                input=text, model=self.model_name
            )
            return list(resp.data[0].embedding)
        raise RuntimeError("Embedding model provider is not properly initialized.")

    def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Directly invokes batch inference on the underlying model provider.
        """
        if self.provider == "local" and self._local_model is not None:
            raw_vecs = self._local_model.encode(texts, normalize_embeddings=True)
            return [[float(x) for x in vec] for vec in raw_vecs]
        elif self.provider == "openai" and self._openai_client is not None:
            resp = self._openai_client.embeddings.create(
                input=texts, model=self.model_name
            )
            # OpenAI preserves index order
            return [list(item.embedding) for item in resp.data]
        raise RuntimeError("Embedding model provider is not properly initialized.")


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Returns cached singleton instance of EmbeddingService.
    """
    return EmbeddingService()
