"""One shared Redis connection, configured from REDIS_URL in .env."""

import os
from functools import lru_cache
from pathlib import Path

import redis
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Aggregate counters, same idea as `langcache:stats` in the Redis tutorial.
STATS_KEY = "cache:stats"


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis:
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError("REDIS_URL is not set. Copy .env.example to .env and fill it in.")
    return redis.Redis.from_url(url, decode_responses=True, socket_timeout=10)


def incr_stat(field: str, amount: int = 1) -> None:
    get_redis().hincrby(STATS_KEY, field, amount)


def read_stats() -> dict[str, int]:
    return {k: int(v) for k, v in get_redis().hgetall(STATS_KEY).items()}


def reset_stats() -> None:
    get_redis().delete(STATS_KEY)
