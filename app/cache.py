import redis
import os
import json

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True
)

def get_conversation(conversation_id: str):
    data = redis_client.get(f"conversation:{conversation_id}")
    return json.loads(data) if data else []

def save_conversation(conversation_id: str, messages: list):
    redis_client.set(
        f"conversation:{conversation_id}",
        json.dumps(messages),
        ex=3600 
    )