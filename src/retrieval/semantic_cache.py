import json
import uuid
from typing import Dict, Optional

from qdrant_client import models
from qdrant_client.models import PointStruct

from src.core.dependencies import get_embeddings, get_qdrant_client, get_redis_client
from src.utils.logger import logger


class SemanticCache:
    def __init__(self, threshold: float = 0.95, collection_name: str = "semantic-cache"):
        self.threshold = threshold
        self.collection_name = collection_name
        self.embeddings = get_embeddings()

        try:
            self.redis = get_redis_client()
        except Exception as e:
            logger.warning(f"SemanticCache: Redis unavailable ({e}). Caching disabled.")
            self.redis = None

        self.qdrant = get_qdrant_client()

    @staticmethod
    def _redis_key(user_id: str, point_id: str) -> str:
        return f"semantic_cache:{user_id}:{point_id}"

    @staticmethod
    def _cache_filter(user_id: str, cache_scope: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="cache_scope",
                    match=models.MatchValue(value=cache_scope),
                ),
            ]
        )

    async def get(
        self, query: str, *, user_id: Optional[str], cache_scope: str = "default"
    ) -> Optional[Dict]:
        """Search Qdrant for a semantically similar query and return its cached response from Redis."""
        if self.redis is None or not user_id:
            return None
        try:
            # 1. Embed the incoming query
            vector = await self.embeddings.aembed_query(query)

            # 2. Search Qdrant for nearest match
            query_response = await self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=self._cache_filter(user_id, cache_scope),
                limit=1,
                score_threshold=self.threshold,
            )
            results = query_response.points

            if not results:
                return None

            # 3. Hit! Get payload from Redis using point ID
            point_id = results[0].id
            cached_data = await self.redis.get(self._redis_key(user_id, str(point_id)))

            if cached_data:
                logger.info(f"Semantic Cache Hit! (score: {results[0].score:.3f})")
                return dict(json.loads(cached_data))

        except Exception as e:
            logger.error(f"SemanticCache get error: {e}")

        return None

    async def set(
        self,
        query: str,
        response_payload: Dict,
        *,
        user_id: Optional[str],
        cache_scope: str = "default",
    ) -> None:
        """Insert query embedding to Qdrant and response payload to Redis."""
        if self.redis is None or not user_id:
            return
        try:
            # 1. Embed the query
            vector = await self.embeddings.aembed_query(query)
            point_id = str(uuid.uuid4())

            # 2. Store payload in Redis
            await self.redis.setex(
                self._redis_key(user_id, point_id),
                86400 * 7,  # 7 days TTL
                json.dumps(response_payload),
            )

            # 3. Store vector in Qdrant
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "query": query,
                            "user_id": user_id,
                            "cache_scope": cache_scope,
                        },
                    )
                ],
            )
            logger.info("Inserted new response into semantic cache.")

        except Exception as e:
            logger.error(f"SemanticCache set error: {e}")

    async def invalidate_user(self, user_id: Optional[str]) -> None:
        """Remove cached answers for one user without touching other Redis data."""
        if self.redis is None or not user_id:
            return
        try:
            if await self.qdrant.collection_exists(self.collection_name):
                await self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id",
                                match=models.MatchValue(value=user_id),
                            )
                        ]
                    ),
                )
            async for key in self.redis.scan_iter(match=f"semantic_cache:{user_id}:*"):
                await self.redis.delete(key)
            logger.info(f"Invalidated semantic cache for user {user_id}.")
        except Exception as e:
            logger.warning(f"SemanticCache invalidation failed for user {user_id}: {e}")
