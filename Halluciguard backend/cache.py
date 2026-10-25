
import os
import redis
import json
import hashlib

from dotenv import load_dotenv
from logger import logger

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    decode_responses=True
)


def check_redis_connection():
    try:
        redis_client.ping()
        logger.info("Redis connection successful")
        return True
    except redis.ConnectionError:
        logger.error("Redis connection failed")
        return False


def set_cache(key, value):
    redis_client.set(key, value)
    logger.info(f"Stored in Redis: {key}")


def get_cache(key):
    value = redis_client.get(key)
    logger.info(f"Retrieved from Redis: {key}")
    return value


def delete_cache(key):
    redis_client.delete(key)
    logger.info(f"Deleted from Redis: {key}")


def make_cache_key(prompt):
    prompt = prompt.strip().lower()

    hash_value = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    return f"hallucination_cache:{hash_value}"


def set_hallucination_cache(prompt, data):
    key = make_cache_key(prompt)

    redis_client.setex(
        key,
        3600,
        json.dumps(data)
    )

    logger.info(f"Hallucination cached: {key}")


def get_hallucination_cache(prompt):
    key = make_cache_key(prompt)

    value = redis_client.get(key)

    if value:
        logger.info(f"Cache HIT: {key}")
        return json.loads(value)

    logger.info(f"Cache MISS: {key}")
    return None


if __name__ == "__main__":

    check_redis_connection()

    set_cache(
        "test_key",
        "Hello HalluciGuard!"
    )

    get_cache("test_key")

    delete_cache("test_key")

    get_cache("test_key")