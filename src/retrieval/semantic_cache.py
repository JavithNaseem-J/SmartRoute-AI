import json
import uuid
from typing import Dict, Optional

from qdrant_client.models import PointStruct

from src.core.dependencies import get_embeddings, get_qdrant_client, get_redis_client
from src.utils.logger import logger


class SemanticCache:
    def __init__(self, threshold: float = 0.95, collection_name: str = "semantic-cache"):
        self.threshold = threshold
        self.collection_name = collection_name
        self.embeddings = get_embeddings()

        self.redis = get_redis_client()

        self.qdrant = get_qdrant_client()

    async def get(self, query: str) -> Optional[Dict]:
        """Search Qdrant for a semantically similar query and return its cached response from Redis."""
        try:
            # 1. Embed the incoming query
            vector = await self.embeddings.aembed_query(query)

            # 2. Search Qdrant for nearest match
            results = await self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=1,
                score_threshold=self.threshold,
            )

            if not results:
                return None

            # 3. Hit! Get payload from Redis using point ID
            point_id = results[0].id
            cached_data = await self.redis.get(f"semantic_cache:{point_id}")

            if cached_data:
                logger.info(f"Semantic Cache Hit! (score: {results[0].score:.3f})")
                return json.loads(cached_data)

        except Exception as e:
            logger.error(f"SemanticCache get error: {e}")

        return None

    async def set(self, query: str, response_payload: Dict) -> None:
        """Insert query embedding to Qdrant and response payload to Redis."""
        try:
            # 1. Embed the query
            vector = await self.embeddings.aembed_query(query)
            point_id = str(uuid.uuid4())

            # 2. Store payload in Redis
            await self.redis.setex(
                f"semantic_cache:{point_id}",
                86400 * 7,  # 7 days TTL
                json.dumps(response_payload),
            )

            # 3. Store vector in Qdrant
            await self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(id=point_id, vector=vector, payload={"query": query})],
            )
            logger.info("Inserted new response into semantic cache.")

        except Exception as e:
            logger.error(f"SemanticCache set error: {e}")
