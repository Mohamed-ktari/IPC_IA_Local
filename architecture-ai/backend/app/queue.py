# queue.py
# RQ (Redis Queue) setup — reuses the same Redis instance qa_agent already
# uses for conversation state, just a different logical use (job queue vs
# key-value store). Single queue for now ("default") — split into multiple
# queues (e.g. "generation", "high_priority") later if job types diversify.

import redis
from rq import Queue
from app.config import settings

_redis_conn = redis.Redis.from_url(settings.REDIS_URL)

generation_queue = Queue("generation", connection=_redis_conn)