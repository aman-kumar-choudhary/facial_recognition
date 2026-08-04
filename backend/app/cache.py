"""
Redis cache layer.

Used for:
  - Caching frequently accessed student info (avoids a DB hit on every auth)
  - Caching recent authentication results by a short-lived key (rate limiting /
    duplicate-frame suppression on the frontend polling loop)
Not used for the embedding similarity search itself -- that's FAISS's job
(see vector_store.py). Redis here is purely a cache in front of the primary
metadata store, matching the architecture doc.
"""
import json
from typing import Optional

import redis.asyncio as redis

from app.config import settings

_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


async def cache_student(student_id: str, payload: dict) -> None:
    r = get_redis()
    await r.set(f"student:{student_id}", json.dumps(payload), ex=settings.REDIS_STUDENT_TTL_SECONDS)


async def get_cached_student(student_id: str) -> Optional[dict]:
    r = get_redis()
    raw = await r.get(f"student:{student_id}")
    return json.loads(raw) if raw else None


async def invalidate_student(student_id: str) -> None:
    r = get_redis()
    await r.delete(f"student:{student_id}")


async def cache_auth_result(request_hash: str, payload: dict) -> None:
    r = get_redis()
    await r.set(
        f"auth_result:{request_hash}", json.dumps(payload), ex=settings.REDIS_AUTH_RESULT_TTL_SECONDS
    )
